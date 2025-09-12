import pytest
import asyncio
import anda_py
from anda_py import PyCommandType, PyAndaDB

import logging
logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_create_success():
    db_config = {
        "store_location_type": "in_mem",
        "store_location": "",
        "db_name": "test_db",
        "db_desc": "Test DB",
        "meta_cache_capacity": 10000
    }
    db = await PyAndaDB.create(db_config)
    assert isinstance(db, PyAndaDB)

@pytest.mark.asyncio
async def test_create_invalid_config():
    db_config = {
        "store_location_type": "Local_file",
        "store_location": "",  # Invalid: required for Local_file
        "db_name": "bad_db"
    }
    with pytest.raises(RuntimeError):
        await PyAndaDB.create(db_config)

@pytest.mark.asyncio
async def test_execute_kip_success():
    db_config = {
        "store_location_type": "in_mem",
        "store_location": "",
        "db_name": "test_db",
        "db_desc": "Test DB",
        "meta_cache_capacity": 10000
    }
    db = await PyAndaDB.create(db_config)
    command = "FIND(?x) WHERE { ?x {type: 'TestType'} }"
    result = await db.execute_kip(command)
    assert isinstance(result, dict)
    assert "type" in result
    assert "response" in result
    # type should be PyCommandType
    assert type(result["type"]).__name__ == "PyCommandType"
    # response should be a dict or list (depending on command)
    assert isinstance(result["response"], dict)

@pytest.mark.asyncio
async def test_execute_kip_invalid_command():
    db_config = {
        "store_location_type": "in_mem",
        "store_location": "",
        "db_name": "test_db",
        "db_desc": "Test DB",
        "meta_cache_capacity": 10000
    }
    db = await PyAndaDB.create(db_config)
    bad_command = "INVALID_COMMAND"
    result = await db.execute_kip(bad_command)
    assert isinstance(result, dict)
    assert "type" in result
    assert "response" in result
    # type should be PyCommandType
    assert type(result["type"]).__name__ == "PyCommandType"
    # response should contain error info (implementation dependent)
    assert isinstance(result["response"], dict)
    # Optionally inspect for error keys/messages
    if isinstance(result["response"], dict):
        assert result["type"] == PyCommandType.Unknown
        assert len(result["response"]) and "error" in result["response"]
