"""Начальная схема: все таблицы из моделей.

Revision ID: 0001
Revises:
"""
from forecast_service import models

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    models.SQLModel.metadata.create_all(op.get_bind())


def downgrade():
    models.SQLModel.metadata.drop_all(op.get_bind())
