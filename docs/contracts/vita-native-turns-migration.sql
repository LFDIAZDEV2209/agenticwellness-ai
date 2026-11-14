BEGIN;

-- Running upgrade 18c4f6a91b20 -> 94e1c2b75a06

ALTER TABLE ai.vita_voice_turns DROP CONSTRAINT uq_vita_voice_turns_session_id;

UPDATE ai.alembic_version SET version_num='94e1c2b75a06' WHERE ai.alembic_version.version_num = '18c4f6a91b20';

COMMIT;
