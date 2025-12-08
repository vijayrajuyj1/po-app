"""Add field response comments, issues, edits tables

Revision ID: 0003
Revises: 0002
Create Date: 2025-11-23 13:05:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade():
    # field_response_comments
    op.create_table(
        'field_response_comments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('field_response_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['field_response_id'], ['field_responses.id'], name=op.f('fk_field_response_comments_field_response_id_field_responses'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['author_id'], ['users.id'], name=op.f('fk_field_response_comments_author_id_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_field_response_comments')),
    )
    op.create_index(op.f('ix_field_response_comments_field_response_id'), 'field_response_comments', ['field_response_id'], unique=False)
    op.create_index(op.f('ix_field_response_comments_author_id'), 'field_response_comments', ['author_id'], unique=False)

    # field_response_issues
    op.create_table(
        'field_response_issues',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('field_response_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('reporter_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('severity', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='open'),
        sa.Column('is_blocking', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(['field_response_id'], ['field_responses.id'], name=op.f('fk_field_response_issues_field_response_id_field_responses'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reporter_id'], ['users.id'], name=op.f('fk_field_response_issues_reporter_id_users'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['resolved_by'], ['users.id'], name=op.f('fk_field_response_issues_resolved_by_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_field_response_issues')),
    )
    op.create_index(op.f('ix_field_response_issues_field_response_id'), 'field_response_issues', ['field_response_id'], unique=False)
    op.create_index(op.f('ix_field_response_issues_reporter_id'), 'field_response_issues', ['reporter_id'], unique=False)

    # field_response_edits
    op.create_table(
        'field_response_edits',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('field_response_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('actor_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(length=30), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('before_answer', sa.Text(), nullable=True),
        sa.Column('before_short_answer', sa.Text(), nullable=True),
        sa.Column('before_citations', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('after_answer', sa.Text(), nullable=True),
        sa.Column('after_short_answer', sa.Text(), nullable=True),
        sa.Column('after_citations', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['field_response_id'], ['field_responses.id'], name=op.f('fk_field_response_edits_field_response_id_field_responses'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['actor_id'], ['users.id'], name=op.f('fk_field_response_edits_actor_id_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_field_response_edits')),
    )
    op.create_index(op.f('ix_field_response_edits_field_response_id'), 'field_response_edits', ['field_response_id'], unique=False)
    op.create_index(op.f('ix_field_response_edits_actor_id'), 'field_response_edits', ['actor_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_field_response_edits_actor_id'), table_name='field_response_edits')
    op.drop_index(op.f('ix_field_response_edits_field_response_id'), table_name='field_response_edits')
    op.drop_table('field_response_edits')

    op.drop_index(op.f('ix_field_response_issues_reporter_id'), table_name='field_response_issues')
    op.drop_index(op.f('ix_field_response_issues_field_response_id'), table_name='field_response_issues')
    op.drop_table('field_response_issues')

    op.drop_index(op.f('ix_field_response_comments_author_id'), table_name='field_response_comments')
    op.drop_index(op.f('ix_field_response_comments_field_response_id'), table_name='field_response_comments')
    op.drop_table('field_response_comments')


