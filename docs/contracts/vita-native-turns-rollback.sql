BEGIN;

-- Running downgrade 94e1c2b75a06 -> 18c4f6a91b20

LOCK TABLE ai.vita_voice_turns IN SHARE ROW EXCLUSIVE MODE;

DO $$ BEGIN
          IF EXISTS (
            SELECT 1 FROM ai.vita_voice_turns
            GROUP BY session_id, provider_turn HAVING count(*) > 1
          ) THEN
            RAISE EXCEPTION 'voice_counter_duplicates: downgrade refused; retain schema and rollback application to legacy voice';
          END IF;
        END $$;

ALTER TABLE ai.vita_voice_turns ADD CONSTRAINT uq_vita_voice_turns_session_id UNIQUE (session_id, provider_turn);

UPDATE ai.alembic_version SET version_num='18c4f6a91b20' WHERE ai.alembic_version.version_num = '94e1c2b75a06';

COMMIT;
