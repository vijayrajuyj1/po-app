"""Add field_responses table

Revision ID: 0002
Revises: 0001
Create Date: 2025-11-23 12:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade():
    # Create field_response_status enum if it doesn't exist
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE field_response_status AS ENUM ('Pending', 'To be verified', 'Processing', 'Processed', 'Verified', 'Failed');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create field_responses table
    op.create_table(
        'field_responses',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('extraction_run_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('category_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('field_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('question', sa.Text(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column('short_answer', sa.Text(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('citations', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='Pending'),
        sa.Column('is_modified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('modified_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('modified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('verified_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_flagged', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('flagged_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('flagged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('flag_details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('answer_history', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),

        # Primary key
        sa.PrimaryKeyConstraint('id', name=op.f('pk_field_responses')),

        # Foreign keys
        sa.ForeignKeyConstraint(['extraction_run_id'], ['extraction_runs.id'], name=op.f('fk_field_responses_extraction_run_id_extraction_runs'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], name=op.f('fk_field_responses_session_id_sessions'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_field_responses_category_id_categories'), ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['field_id'], ['fields.id'], name=op.f('fk_field_responses_field_id_fields'), ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['modified_by'], ['users.id'], name=op.f('fk_field_responses_modified_by_users'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['verified_by'], ['users.id'], name=op.f('fk_field_responses_verified_by_users'), ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['flagged_by'], ['users.id'], name=op.f('fk_field_responses_flagged_by_users'), ondelete='SET NULL'),

        # Unique constraint
        sa.UniqueConstraint('extraction_run_id', 'field_id', name=op.f('uq_field_responses_extraction_run_id_field_id'))
    )

    # Create indexes
    op.create_index(op.f('ix_field_responses_id'), 'field_responses', ['id'], unique=False)
    op.create_index(op.f('ix_field_responses_extraction_run_id'), 'field_responses', ['extraction_run_id'], unique=False)
    op.create_index(op.f('ix_field_responses_session_id'), 'field_responses', ['session_id'], unique=False)
    op.create_index(op.f('ix_field_responses_category_id'), 'field_responses', ['category_id'], unique=False)
    op.create_index(op.f('ix_field_responses_field_id'), 'field_responses', ['field_id'], unique=False)
    op.create_index(op.f('ix_field_responses_status'), 'field_responses', ['status'], unique=False)
    op.create_index(op.f('ix_field_responses_modified_by'), 'field_responses', ['modified_by'], unique=False)
    op.create_index(op.f('ix_field_responses_verified_by'), 'field_responses', ['verified_by'], unique=False)
    op.create_index(op.f('ix_field_responses_flagged_by'), 'field_responses', ['flagged_by'], unique=False)

    # Create partial index for flagged items
    op.create_index(
        'ix_field_responses_is_flagged',
        'field_responses',
        ['is_flagged'],
        unique=False,
        postgresql_where=sa.text('is_flagged = true')
    )

    # Create composite index for session + category queries
    op.create_index(
        'ix_field_responses_session_category',
        'field_responses',
        ['session_id', 'category_id'],
        unique=False
    )

    # Alter status column to use the enum type
    op.execute("""
        ALTER TABLE field_responses
        ALTER COLUMN status DROP DEFAULT,
        ALTER COLUMN status TYPE field_response_status
        USING status::field_response_status,
        ALTER COLUMN status SET DEFAULT 'Pending'::field_response_status
    """)


def downgrade():
    # Drop table and indexes (indexes are dropped automatically with table)
    op.drop_table('field_responses')

    # Drop enum type
    op.execute('DROP TYPE field_response_status')
