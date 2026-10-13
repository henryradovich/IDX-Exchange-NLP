import json, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from query_parser import QueryParser, SchemaValidator

@pytest.fixture
def parser():
    return QueryParser("data/processed/taxonomy.json")

@pytest.mark.parametrize("q,k,v", [
("homes under 700k","price_max",700000),("below $500k","price_max",500000),
("less than 1.2m","price_max",1200000),("homes over 400k","price_min",400000),
("above $650k","price_min",650000),("more than 1m","price_min",1000000),
("3 bed","bedrooms",3),("4 bedrooms","bedrooms",4),("2 br","bedrooms",2),
("3+ beds","bedrooms_min",3),("at least 4 bedrooms","bedrooms_min",4),
("up to 5 beds","bedrooms_max",5),("no more than 2 br","bedrooms_max",2),
("2 bath","bathrooms",2.0),("2.5 bathrooms","bathrooms",2.5),("3 ba","bathrooms",3.0),
("2+ baths","bathrooms_min",2.0),("at least 2.5 bathrooms","bathrooms_min",2.5),
("up to 3 baths","bathrooms_max",3.0),("no more than 1.5 baths","bathrooms_max",1.5),
])
def test_single_patterns(parser,q,k,v):
    assert parser.parse(q)[k] == v

@pytest.mark.parametrize("q,a,b", [
("between 400k and 600k",400000,600000),
("between $1m and $1.5m",1000000,1500000),
("$500k-$750k",500000,750000),
("500k to 900k",500000,900000),
])
def test_price_ranges(parser,q,a,b):
    out=parser.parse(q); assert out["price_min"]==a and out["price_max"]==b

@pytest.mark.parametrize("q,city", [
("3 bed in Irvine under 700k","Irvine"),
("homes in Portland","Portland"),
("2 bath in San Diego with pool","San Diego"),
("in Los Angeles under 1m","Los Angeles"),
])
def test_city(parser,q,city):
    assert parser.parse(q)["city"] == city

@pytest.mark.parametrize("q,status", [
("show active listings","active"),("find pending homes","pending"),
("which properties are under contract","under_contract"),
("show contingent properties","contingent"),("find recently sold homes","sold"),
("show new listings","new_listing"),("homes back on the market","back_on_market"),
])
def test_status(parser,q,status):
    assert parser.parse(q)["listing_status"] == status

def test_parameterized_sql(parser):
    sql,params=parser.to_sql({"bedrooms":3,"price_max":700000,"city":"Irvine"})
    assert "Irvine" not in sql and "700000" not in sql
    assert params == [700000,3,"Irvine"]

@pytest.mark.parametrize("payload",[
"Irvine' OR 1=1 --","'; DROP TABLE rets_property; --",
"x' UNION SELECT * FROM users --",'\" OR \"1\"=\"1'
])
def test_injection(parser,payload):
    sql,params=parser.to_sql({"city":payload})
    assert payload not in sql and payload in params
    assert "DROP TABLE" not in sql and "UNION SELECT" not in sql

@pytest.mark.parametrize("filters",[
{"price_max":99999},{"price_max":10000001},{"bedrooms":11},
{"bedrooms":-1},{"bathrooms":12},{"price_min":900000,"price_max":500000}
])
def test_invalid_ranges(tmp_path,filters):
    s={"valid_cities":["Irvine","Portland"],"ranges":{"price_min":100000,"price_max":10000000,"bedrooms_max":10,"bathrooms_max":10}}
    p=tmp_path/"schema.json"; p.write_text(json.dumps(s))
    valid,errors=SchemaValidator(str(p)).validate_query(filters)
    assert not valid and errors

def test_invalid_city(tmp_path):
    s={"valid_cities":["Irvine","Portland"],"ranges":{"price_min":100000,"price_max":10000000,"bedrooms_max":10,"bathrooms_max":10}}
    p=tmp_path/"schema.json"; p.write_text(json.dumps(s))
    valid,errors=SchemaValidator(str(p)).validate_query({"city":"Atlantis"})
    assert not valid and "not found" in errors[0]
