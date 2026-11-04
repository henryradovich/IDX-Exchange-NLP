import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from signal_extractor import SignalExtractor

TAX = {'terms': [
    {'term':'pool','category':'Exterior'},
    {'term':'community pool','category':'Amenities'},
    {'term':'mountain views','category':'Location'},
    {'term':'updated','category':'Condition'},
    {'term':'seller financing','category':'Pricing & Transaction'},
    {'term':'stainless steel appliances','category':'Property Features'},
    {'term':'stainless steel','category':'Property Features'},
]}

def test_signals_and_structured():
    e = SignalExtractor(TAX)
    out = e.extract_signals({'L_ListingID':17,'L_City':'Irvine','L_Keyword2':3,
        'L_Remarks':'Updated with community pool, mountain views, and seller financing.'})
    assert out['structured']['bedrooms'] == 3
    assert out['amenities'] == ['community pool']
    assert out['condition_keywords'] == ['updated']
    assert out['location_features'] == ['mountain views']
    assert out['financing_terms'] == ['seller financing']

def test_negation_and_repetition():
    out = SignalExtractor(TAX).extract_signals({'remarks':'No pool. A community pool nearby. Pool access.'})
    assert 'community pool' in out['amenities']
    assert any(m['negated'] for m in out['matches'])
    assert len([m for m in out['matches'] if m['term']=='pool']) == 2

def test_nested_longest():
    out = SignalExtractor(TAX).extract_signals({'remarks':'Stainless-steel appliances in the kitchen.'})
    assert out['amenities'] == ['stainless steel appliances']
    assert out['matches'][0]['text'] == 'Stainless-steel appliances'

def test_without_entity_extractor():
    assert SignalExtractor(TAX).extract_signals({'remarks':''})['entities'] == {}
