import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
from uuid import UUID

import loguru
import pydantic
from sqlmodel import Session
from oasst_backend.api.deps import create_api_client
from oasst_backend.models import ApiClient, Message, MessageTreeState
from oasst_backend.models.message_tree_state import State as TreeState
from oasst_backend.models.payload_column_type import PayloadContainer
from oasst_backend.prompt_repository import PromptRepository
from oasst_backend.user_repository import UserRepository
from oasst_data import ExportMessageNode, ExportMessageTree
import oasst_backend.models.db_payload as db_payload
import oasst_backend.utils.database_utils as db_utils

# Well known id for the API client
IMPORT_API_CLIENT_ID = UUID("bd8fde8b-1d8e-4e9a-9966-e96d000f8363")

class Importer:
    def __init__(self, db: Session, origin: str, model_name: Optional[str] = None):
        self.db = db
        self.origin = origin
        self.model_name = model_name

        # Get import API client
        api_client = db.query(ApiClient).filter(ApiClient.id == IMPORT_API_CLIENT_ID).first()
        if not api_client:
            api_client = create_api_client(
                session=db,
                description="API client used for importing data",
                frontend_type="import",
                force_id=IMPORT_API_CLIENT_ID,
            )

        ur = UserRepository(db, api_client)
        self.import_user = ur.lookup_system_user(username="import")
        self.pr = PromptRepository(db=db, api_client=api_client, user_repository=ur)
        self.api_client = api_client

    def fetch_message(self, message_id: UUID) -> Optional[Message]:
        return self.db.query(Message).filter(Message.id == message_id).one_or_none()

    def fetch_message_tree_state(self, message_tree_id: UUID) -> Optional[MessageTreeState]:
        return self.db.query(MessageTreeState).filter(MessageTreeState.message_tree_id == message_tree_id).one_or_none()

    def import_message(self, message: ExportMessageNode, message_tree_id: UUID, parent_id: Optional[UUID] = None) -> Message:
        payload = db_payload.MessagePayload(text=message.text)
        msg = Message(
            id=message.message_id,
            message_tree_id=message_tree_id,
            frontend_message_id=message.message_id,
            parent_id=parent_id,
            review_count=message.review_count or 0,
            lang=message.lang or "en",
            review_result=True,
            synthetic=message.synthetic if message.synthetic is not None else True,
            model_name=message.model_name or self.model_name,
            role=message.role,
            api_client_id=self.api_client.id,
            payload_type=type(payload).__name__,
            payload=PayloadContainer(payload=payload),
            user_id=self.import_user.id,
        )
        self.db.add(msg)
        self.db.flush()  # Ensure changes are committed to the database
        if parent_id is None:
            self.pr.update_children_counts(msg.id)
        self.db.refresh(msg)
        if message.replies:
            for r in message.replies:
                self.import_message(r, message_tree_id=message_tree_id, parent_id=msg.id)
        return msg

    def import_tree(self, tree: ExportMessageTree, state: TreeState = TreeState.BACKLOG_RANKING) -> Tuple[MessageTreeState, Message]:
        if not tree.message_tree_id or tree.message_tree_id != tree.prompt.message_id:
            raise ValueError("Invalid message tree ID or mismatch with prompt ID")

        root_msg = self.import_message(tree.prompt, message_tree_id=tree.prompt.message_id)
        if state not in (TreeState.BACKLOG_RANKING, TreeState.RANKING):
            raise ValueError(f"{state} not supported for import")

        active = state == TreeState.RANKING
        mts = MessageTreeState(
            message_tree_id=root_msg.id,
            goal_tree_size=0,
            max_depth=0,
            max_children_count=0,
            state=state,
            origin=self.origin,
            active=active,
            lang=root_msg.lang or "en",
        )
        self.db.add(mts)
        self.db.flush()  # Ensure changes are committed to the database
        return mts, root_msg

def handle_json_line(line: str, importer: Importer, num_activate: int, max_count: Optional[int], imported_count: int) -> int:
    try:
        dict_node = json.loads(line)
        if dict_node.get("message_tree_id"):  # Handle tree
            tree = pydantic.parse_obj_as(ExportMessageTree, dict_node)
            existing_mts = importer.fetch_message_tree_state(tree.message_tree_id)
            if existing_mts:
                loguru.logger.info(f"Skipping existing message tree: {tree.message_tree_id}")
            else:
                state = TreeState.BACKLOG_RANKING if imported_count >= num_activate else TreeState.RANKING
                mts, root_msg = importer.import_tree(tree, state=state)
                imported_count += 1
                loguru.logger.info(
                    f"Imported tree: {mts.message_tree_id}, {mts.state=}, {mts.active=}, {root_msg.children_count=}"
                )
            if max_count and imported_count >= max_count:
                loguru.logger.info(f"Reached max count {max_count} of trees to import.")
                return imported_count
        elif dict_node.get("message_id"):  # Handle message
            message = pydantic.parse_obj_as(ExportMessageNode, dict_node)
            existing_msg = importer.fetch_message(message.message_id)
            if existing_msg:
                loguru.logger.info(f"Skipping existing message: {message.message_id}")
            else:
                msg = importer.import_message(message, message_tree_id=message.message_id)
                imported_count += 1
                loguru.logger.info(f"Imported message: {msg.id}")
    except json.JSONDecodeError as e:
        loguru.logger.error(f"Failed to decode JSON line: {line.strip()}. Error: {e}")
    except pydantic.ValidationError as e:
        loguru.logger.error(f"Validation error for line: {line.strip()}. Error: {e}")
    except Exception as e:
        loguru.logger.error(f"Unexpected error: {e}")

    return imported_count

def import_file(
    input_file_path: Path,
    origin: str,
    *,
    model_name: Optional[str] = None,
    num_activate: int = 0,
    max_count: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    @db_utils.managed_tx_function(auto_commit=db_utils.CommitMode.ROLLBACK if dry_run else db_utils.CommitMode.COMMIT)
    def import_tx(db: Session) -> int:
        importer = Importer(db, origin=origin, model_name=model_name)
        imported_count = 0

        with input_file_path.open() as file_in:
            for line in file_in:
                imported_count = handle_json_line(line, importer, num_activate, max_count, imported_count)
                if max_count and imported_count >= max_count:
                    break

        return imported_count

    if dry_run:
        loguru.logger.info("DRY RUN with rollback")
    return import_tx()

def parse_args() -> argparse.Namespace:
    def str2bool(v: str) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_path", help="Path to the input file")
    parser.add_argument("--origin", type=str, default=None, help="Origin of message trees")
    parser.add_argument("--model_name", type=str, default=None, help="Default model name (if missing in messages)")
    parser.add_argument("--num_activate", type=int, default=0, help="Number of trees to set in ranking state")
    parser.add_argument("--max_count", type=int, default=None, help="Maximum number of message trees to import")
    parser.add_argument("--dry_run", type=str2bool, default=False, help="Run without committing changes")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    input_file_path = Path(args.input_file_path)
    if not input_file_path.exists() or not input_file_path.is_file():
        loguru.logger.error(f"Invalid input file: {args.input_file_path}")
        sys.exit(1)

    num_imported = import_file(
        input_file_path,
        origin=args.origin or input_file_path.name,
        model_name=args.model_name,
        num_activate=args.num_activate,
        max_count=args.max_count,
        dry_run=args.dry_run,
    )
    loguru.logger.info(f"Imported {num_imported} messages and trees.")

if __name__ == "__main__":
    main()
