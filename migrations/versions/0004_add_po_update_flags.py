"""Add po_update_flags table

Revision ID: 0004
Revises: 0003
Create Date: 2025-11-23 13:22:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'po_update_flags',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('extraction_run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='open'),
        sa.Column('flagged_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('admin_note', sa.Text(), nullable=True),
        sa.Column('resolved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], name=op.f('fk_po_update_flags_session_id_sessions'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['extraction_run_id'], ['extraction_runs.id'], name=op.f('fk_po_update_flags_extraction_run_id_extraction_runs'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['flagged_by'], ['users.id'], name=op.f('fk_po_update_flags_flagged_by_users'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['resolved_by'], ['users.id'], name=op.f('fk_po_update_flags_resolved_by_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_po_update_flags')),
    )
    op.create_index(op.f('ix_flag_run'), 'po_update_flags', ['extraction_run_id'], unique=False)
    op.create_index(op.f('ix_flag_session'), 'po_update_flags', ['session_id'], unique=False)
    op.create_index(op.f('ix_flag_status'), 'po_update_flags', ['status'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_flag_status'), table_name='po_update_flags')
    op.drop_index(op.f('ix_flag_session'), table_name='po_update_flags')
    op.drop_index(op.f('ix_flag_run'), table_name='po_update_flags')
    op.drop_table('po_update_flags')


