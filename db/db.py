import sqlite3
import os
import json
import uuid
import datetime
from pathlib import Path

# Module-level variable for configured DB path
_DB_PATH = None


def get_db_path():
    """Returns the configured database path."""
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = os.path.join(os.getcwd(), "beatbunny.db")
    return _DB_PATH


def init_db(db_path=None):
    """
    Creates DB if missing and runs schema.sql.
    Sets the module-level DB path for all subsequent operations.
    """
    global _DB_PATH
    
    if db_path is None:
        db_path = get_db_path()
    else:
        _DB_PATH = db_path  # Store the configured path
    
    # Ensure db directory exists if path contains directories
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema_sql = f.read()
            cursor.executescript(schema_sql)
    else:
        print(f"Warning: schema.sql not found at {schema_path}")

    conn.commit()
    conn.close()


def create_job(lyrics, tags, params):
    """
    Inserts a new job, returns job_id.
    params is a dict, stored as JSON.
    """
    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status = "pending"
    params_json = json.dumps(params)

    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO jobs (id, created_at, status, lyrics, tags, params)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (job_id, created_at, status, lyrics, tags, params_json),
    )

    conn.commit()
    conn.close()

    return job_id


def update_job_status(job_id, status, fields=None):
    """
    Updates job status and optionally other fields (passed as dict).
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    query = "UPDATE jobs SET status = ?"
    values = [status]

    if fields:
        for key, value in fields.items():
            query += f", {key} = ?"
            values.append(value)

    query += " WHERE id = ?"
    values.append(job_id)

    cursor.execute(query, values)
    conn.commit()
    conn.close()


def add_artifact(job_id, artifact_type, path):
    """
    Adds an artifact record for a job.
    artifact_type: 'audio_wav', 'audio_mp3', 'metadata_json'
    """
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO artifacts (job_id, type, path, created_at)
        VALUES (?, ?, ?, ?)
    """,
        (job_id, artifact_type, path, created_at),
    )

    conn.commit()
    conn.close()


def get_recent_jobs(limit=20):
    """
    Returns list of jobs for history.
    """
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM jobs 
        ORDER BY created_at DESC 
        LIMIT ?
    """,
        (limit,),
    )

    jobs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jobs


def get_job(job_id):
    """
    Returns job row as dict.
    """
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return dict(row)
    return None


def get_job_artifacts(job_id):
    """
    Returns artifacts rows as list of dicts.
    """
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM artifacts WHERE job_id = ?", (job_id,))
    artifacts = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return artifacts


if __name__ == "__main__":
    # Minimal sanity checks
    print("Running DB sanity checks...")

    # 1. Init DB
    test_db_path = get_db_path()
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    init_db()
    assert os.path.exists(test_db_path), "Database file should exist after init"
    print("✓ Database initialized")

    # 2. Create Job
    test_lyrics = "Test lyrics"
    test_tags = "rock, test"
    test_params = {"cfg": 7.5, "length": 30}
    job_id = create_job(test_lyrics, test_tags, test_params)
    assert job_id is not None, "Job ID should be returned"
    print(f"✓ Job created: {job_id}")

    # 3. Get Job
    job = get_job(job_id)
    assert job is not None
    assert job["id"] == job_id
    assert job["status"] == "pending"
    assert job["lyrics"] == test_lyrics
    assert json.loads(job["params"]) == test_params
    print("✓ Job retrieval verified")

    # 4. Update Job
    update_job_status(job_id, "processing")
    job = get_job(job_id)
    assert job is not None
    assert job["status"] == "processing"
    print("✓ Job status update verified")

    # 5. Add Artifact
    test_path = "/tmp/test.wav"
    add_artifact(job_id, "audio_wav", test_path)
    artifacts = get_job_artifacts(job_id)
    assert len(artifacts) == 1
    assert artifacts[0]["path"] == test_path
    assert artifacts[0]["type"] == "audio_wav"
    print("✓ Artifact addition verified")

    # 6. Recent Jobs
    jobs = get_recent_jobs()
    assert len(jobs) >= 1
    print("✓ Recent jobs list verified")

    print("All sanity checks passed.")
