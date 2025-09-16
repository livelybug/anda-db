import pytest
import asyncio
import anda_py
from anda_py import PyCommandType, PyAndaDB, StoreLocationType, AndaDbConfig

@pytest.mark.asyncio
async def test_create_success():
    # db_config = {
    #     "store_location_type": StoreLocationType.InMem,
    #     "store_location": "",
    #     "db_name": "test_db",
    #     "db_desc": "Test DB",
    #     "meta_cache_capacity": 10000
    # }
    db_config = AndaDbConfig(
      StoreLocationType.InMem, 
      "", 
      "test_db"
    )
    db = await PyAndaDB.create(db_config)
    assert isinstance(db, PyAndaDB)

@pytest.mark.asyncio
async def test_create_invalid_config():
    db_config = AndaDbConfig(
        StoreLocationType.LocalFile,
        "",  # Invalid: required for Local_file
        "bad_db"
    )
    with pytest.raises(RuntimeError):
        await PyAndaDB.create(db_config)

@pytest.mark.asyncio
async def test_execute_kip_success():
    db_config = AndaDbConfig(
      StoreLocationType.InMem, 
      "", 
      "test_db",
      "Test_DB",
      10000
    )
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
    db_config = AndaDbConfig(
      StoreLocationType.InMem, 
      "", 
      "test_db",
      "Test_DB",
      10000
    )
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
