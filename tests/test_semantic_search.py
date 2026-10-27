import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from semantic_search import SemanticSearcher, BM25Searcher, tokenize
from evaluate_search import evaluate_query

class FakeModel:
    def encode(self, texts, **kwargs):
        rows=[]
        for t in texts:
            words=tokenize(t)
            rows.append([float('pool' in words),float('mountain' in words),float('kitchen' in words)])
        return np.array(rows,dtype=np.float32)

@pytest.fixture
def corpus():
    return ['pool and kitchen','mountain views','modern kitchen'],['a','b','c']

def test_tokenize():
    assert tokenize('Mountain-Views!') == ['mountain','views']

def test_empty_index_raises():
    with pytest.raises(RuntimeError): SemanticSearcher(model=FakeModel()).search('pool')

def test_duplicate_ids(corpus):
    pytest.importorskip('faiss')
    with pytest.raises(ValueError): SemanticSearcher(model=FakeModel()).build_index(corpus[0],['a','a','c'])

def test_semantic_search(corpus):
    pytest.importorskip('faiss')
    s=SemanticSearcher(model=FakeModel()).build_index(*corpus)
    assert s.search('mountain',1)[0]['id']=='b'
    assert len(s.search('mountain',99))==3

def test_semantic_roundtrip(corpus,tmp_path):
    pytest.importorskip('faiss')
    s=SemanticSearcher(model=FakeModel()).build_index(*corpus)
    s.save(tmp_path)
    s2=SemanticSearcher(model=FakeModel()).load(tmp_path)
    assert s2.search('mountain',1)[0]['id']=='b'

def test_bm25(corpus):
    pytest.importorskip('rank_bm25')
    s=BM25Searcher().build_index(*corpus)
    assert s.search('mountain',1)[0]['id']=='b'

def test_metrics(corpus):
    class Dummy:
        def search(self,q,top_k=10): return [{'id':'b'},{'id':'a'}]
    metrics=evaluate_query(Dummy(),'mountain',{'b':2,'c':1},top_k=2)
    assert metrics['precision_at_k']==0.5
    assert metrics['recall_at_k']==0.5
    assert 0 < metrics['ndcg_at_k'] <= 1
