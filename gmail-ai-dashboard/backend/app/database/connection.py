from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.email import Base


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(
	bind=engine,
	autoflush=False,
	autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()


def init_db() -> None:
	"""Create the schema and add columns introduced after the initial setup.

	SQLAlchemy's ``create_all`` does not alter existing tables, so deployments
	with an older database need this small, idempotent schema reconciliation.
	"""
	Base.metadata.create_all(bind=engine)
	inspector = inspect(engine)
	with engine.begin() as connection:
		for table in Base.metadata.sorted_tables:
			if table.name not in inspector.get_table_names():
				continue

			existing_columns = {
				column["name"] for column in inspector.get_columns(table.name)
			}
			for column in table.columns:
				if column.name in existing_columns:
					continue

				column_type = column.type.compile(dialect=engine.dialect)
				quoted_table = engine.dialect.identifier_preparer.quote(table.name)
				quoted_column = engine.dialect.identifier_preparer.quote(column.name)
				connection.execute(
					text(
						f"ALTER TABLE {quoted_table} ADD COLUMN "
						f"{quoted_column} {column_type}"
					)
				)
