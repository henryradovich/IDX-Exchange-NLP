from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
import pytest
from query_intent_classifier import QueryIntentClassifier, LABELS
from understand_query import understand

@pytest.fixture
def clf():
    q=['show me homes','browse houses in Irvine','compare HOA fees','research housing taxes',
       'schedule a tour tomorrow','book a showing this weekend']
    y=['browsing','browsing','researching','researching','high_intent_inquiry','high_intent_inquiry']
    return QueryIntentClassifier().train(q,y)

def test_probability_sums_to_one(clf):
    p=clf.predict_proba('book a showing')
    assert set(p)==set(LABELS)
    assert abs(sum(p.values())-1)<1e-7

def test_predict_contract(clf):
    label,confidence=clf.predict('browse homes in Irvine')
    assert label in LABELS and 0<=confidence<=1

def test_uncertainty_output(clf):
    x=clf.analyze('show me homes',uncertainty_threshold=1.0)
    assert x['uncertain'] and 'probabilities' in x

def test_untrained_raises():
    with pytest.raises(RuntimeError):QueryIntentClassifier().predict('hi')

def test_invalid_labels():
    with pytest.raises(ValueError):QueryIntentClassifier().train(['a','b'],['browsing','price'])

def test_persistence(tmp_path,clf):
    f=tmp_path/'model.joblib';clf.save(f)
    restored=QueryIntentClassifier.load(f)
    assert restored.predict('research housing taxes')[0]==clf.predict('research housing taxes')[0]

def test_parser_integration(tmp_path,clf):
    f=tmp_path/'model.joblib';clf.save(f)
    class StubParser:
        def parse(self,q):return {'price_max':900000}
    r=understand('homes under 900k',str(f),parser=StubParser())
    assert r['filters']=={'price_max':900000}
