import pandas as pd
import pytest
from src.model import risk_band
from src.talk_to_data import validate_sql, ask

def test_risk_bands():
    assert risk_band(.02)=="Low" and risk_band(.15)=="Medium" and risk_band(.4)=="High"

def test_sql_guard_blocks_delete():
    with pytest.raises(ValueError): validate_sql("DELETE FROM applications")

def test_fallback_query_executes():
    df=pd.DataFrame({"TARGET":[0,1,0],"CODE_GENDER":["F","M","F"]})
    sql,result=ask(df,"overall default rate")
    assert "SELECT" in sql and len(result)==1

