#!/usr/bin/env python3
"""Rule-based real-estate query parsing with parameterized MySQL SQL generation.

Run from project root. `scripts/` is NOT required to be a Python package.
"""
import json
import re
from pathlib import Path

NUMBER_WORDS = dict(zip('one two three four five six seven eight nine ten'.split(), range(1, 11)))
SYNONYMS = {
    'newly constructed': 'new construction', 'brand new home': 'new construction',
    'fully renovated': 'fully remodeled', 'recently renovated': 'recently remodeled',
    'need updating': 'needs updating', 'fixer-uppers': 'fixer upper',
    'fixer-upper': 'fixer upper', 'close to parks': 'near parks',
    'nearby parks': 'near parks', 'finished basement': 'finished basement',
    'outdoor living space': 'outdoor living', 'cul-de-sac': 'cul de sac',
    'near shopping and restaurants': 'near shopping',
    'well-maintained': 'well maintained', 'move-in ready': 'move in ready',
}
# Ambiguous unqualified fragments frequently matched as parts of richer phrases.
OMIT_GENERIC = {'bedroom', 'bedrooms', 'bathroom', 'bathrooms', 'home', 'garage',
                'office', 'kitchen', 'hoa', 'hoa fees', 'pending', 'sold', 'contingent',
                'sq ft', 'square feet', 'close to',
                'built in', 'floor plan', 'laundry', 'garden'}


def _phrase_pattern(phrase):
    words = re.split(r'[\s-]+', phrase)
    return re.compile(r'(?<!\w)' + r'[\s-]+'.join(map(re.escape, words)) + r'(?!\w)', re.I)


class QueryParser:
    def __init__(self, taxonomy_path='data/processed/taxonomy.json', schema_path='data/schema.json'):
        self.taxonomy_path = Path(taxonomy_path)
        self.schema_path = Path(schema_path)
        self.amenity_terms = self._load_terms()
        if self.schema_path.exists():
            schema = json.loads(self.schema_path.read_text(encoding='utf-8'))
            self.valid_cities = sorted(set(schema.get('valid_cities', [])), key=len, reverse=True)
        else:
            self.valid_cities = []

    def _load_terms(self):
        if not self.taxonomy_path.exists():
            return []
        data = json.loads(self.taxonomy_path.read_text(encoding='utf-8'))
        categories = {'amenities','property features','exterior','location','interior','rooms',
                      'condition','pricing & transaction'}
        terms = [str(t['term']).strip().lower() for t in data.get('terms', [])
                 if t.get('category','').lower() in categories and t.get('term')]
        return sorted(set(terms) - OMIT_GENERIC, key=lambda x: (-len(x), x))

    @staticmethod
    def _num(value, suffix=''):
        n = float(value.replace(',', ''))
        return int(n * {'k':1000, 'm':1000000}.get(suffix.lower(), 1))

    @staticmethod
    def _normalize_number_words(text):
        for word, value in NUMBER_WORDS.items():
            text = re.sub(rf'\b{word}(?=[\s-]+(?:bedrooms?|beds?|br|bd|bathrooms?|baths?|ba)\b)',
                          str(value), text, flags=re.I)
        return text

    def parse(self, query):
        if not isinstance(query, str):
            raise TypeError('query must be a string')
        s = self._normalize_number_words(query.lower())
        filters = {}
        numeric = r'(\d[\d,]*(?:\.\d+)?)\s*([km]?)'

        for expression, key in [
            (rf'\b(?:under|below|less than|at most|max(?:imum)?(?: price)?)\s+\$?{numeric}\b','price_max'),
            (rf'\b(?:over|above|more than|at least|min(?:imum)?(?: price)?)\s+\$?{numeric}\b','price_min'),
        ]:
            match = re.search(expression, s)
            if match:
                filters[key] = self._num(*match.groups())
        between = re.search(rf'\bbetween\s+\$?{numeric}\s+(?:and|to)\s+\$?{numeric}\b', s)
        if between:
            a,b,c,d = between.groups()
            filters.update(price_min=self._num(a,b), price_max=self._num(c,d))
        compact = re.search(rf'(?<!\w)\$?{numeric}\s*(?:-|to)\s*\$?{numeric}\b', s)
        if compact and not ('price_min' in filters or 'price_max' in filters):
            a,b,c,d = compact.groups()
            filters.update(price_min=self._num(a,b), price_max=self._num(c,d))

        for kind, noun, cast in [('bedrooms',r'(?:bedrooms?|beds?|br|bd)',int),
                                 ('bathrooms',r'(?:bathrooms?|baths?|ba)',float)]:
            num = r'(\d+(?:\.\d+)?)'
            rules = [
                (rf'\b(?:at least|minimum|min)\s+{num}[\s-]*{noun}\b',kind+'_min'),
                (rf'\b{num}\s*\+[\s-]*{noun}\b',kind+'_min'),
                (rf'\b(?:up to|no more than|at most|maximum|max)\s+{num}[\s-]*{noun}\b',kind+'_max'),
                (rf'\b{num}[\s-]*{noun}\b',kind),
            ]
            for pattern,key in rules:
                match = re.search(pattern,s)
                if match:
                    filters[key] = cast(match.group(1))
                    break

        # Known-city matching only; avoid identifying 'in a quiet neighborhood' as a city.
        for city in self.valid_cities:
            if re.search(r'(?<!\w)in\s+' + re.escape(city) + r'(?!\w)', s, re.I):
                filters['city'] = city
                break
        # Without a city list, extract explicit city candidates only when they look
        # like proper names. Validator must check them against database values.
        if 'city' not in filters:
            m = re.search(r'(?<!\w)in\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3})(?=\b)', query)
            if m:
                filters['city'] = m.group(1)

        statuses = {
            'active':r'\b(?:active listings?|still available|currently available|available homes?)\b',
            'pending':r'\bpending\b', 'under_contract':r'\bunder contract\b',
            'contingent':r'\bcontingent\b', 'sold':r'\b(?:recently )?sold\b',
            'new_listing':r'\bnew listings?\b',
            'back_on_market':r'\b(?:back|came back) on (?:the )?market\b',
        }
        for value, pat in statuses.items():
            if re.search(pat, s):
                filters['listing_status'] = value
                break
        if re.search(r'\b(?:least expensive|cheapest|lowest priced)\b', s):
            filters['sort'] = 'price_asc'
        elif re.search(r'\b(?:most expensive|highest priced)\b', s):
            filters['sort'] = 'price_desc'
        if re.search(r'\b(?:no hoa|without (?:an? )?hoa|hoa[ -]free)\b', s):
            filters['hoa'] = False
        elif re.search(r'\b(?:low|minimal) hoa fees?\b', s):
            filters['hoa_fee_max_preference'] = 'low'

        # Handle coordinated location features before matching individual spans.
        if re.search(r'\bnear\s+shopping\s+and\s+restaurants\b', s):
            coordinated_location = {'near shopping', 'near restaurants'}
        else:
            coordinated_location = set()

        # Collect candidate occurrences (including synonyms), prioritize longest spans.
        candidates = []
        vocabulary = [(term,term) for term in self.amenity_terms]
        vocabulary += list(SYNONYMS.items())
        for surface, canonical in vocabulary:
            for m in _phrase_pattern(surface).finditer(s):
                # Suppress HOA as a generic amenity; handled by separate rules.
                if canonical == 'hoa':
                    continue
                candidates.append((m.start(),m.end(),canonical))
        candidates.sort(key=lambda x: (-(x[1]-x[0]),x[0]))
        occupied = []
        included,excluded = set(coordinated_location),set()
        for start,end,term in candidates:
            if any(start < b and end > a for a,b in occupied):
                continue
            occupied.append((start,end))
            context = s[max(0,start-45):start]
            negated = bool(re.search(r'\b(?:without|no|not|exclude|excluding)\s+(?:(?:a|an|any|the)\s+)?$',context))
            (excluded if negated else included).add(term)
        if included:
            filters['amenities_include'] = sorted(included)
        if excluded:
            filters['amenities_exclude'] = sorted(excluded)
        return filters

    def to_sql(self, filters):
        """Build parameterized SQL; reject unsupported keys rather than ignoring them."""
        supported = {'price_min','price_max','bedrooms','bedrooms_min','bedrooms_max',
                     'bathrooms','bathrooms_min','bathrooms_max','city',
                     'amenities_include','amenities_exclude','sort'}
        unknown = set(filters) - supported
        if unknown:
            raise ValueError('Filters not supported by current SQL schema: ' + ', '.join(sorted(unknown)))
        clauses,params = [],[]
        columns = {
            'price_min':('L_SystemPrice','>='),'price_max':('L_SystemPrice','<='),
            'bedrooms':('L_Keyword2','='),'bedrooms_min':('L_Keyword2','>='),
            'bedrooms_max':('L_Keyword2','<='),
            'bathrooms':('LM_Dec_3','='),'bathrooms_min':('LM_Dec_3','>='),
            'bathrooms_max':('LM_Dec_3','<='),'city':('L_City','='),
        }
        for k,(col,op) in columns.items():
            if k in filters:
                clauses.append(f'{col} {op} %s')
                params.append(filters[k])
        for op,key in [('LIKE','amenities_include'),('NOT LIKE','amenities_exclude')]:
            for term in filters.get(key,[]):
                escaped = term.lower().replace('\\','\\\\').replace('%',r'\%').replace('_',r'\_')
                clauses.append(f"LOWER(L_Remarks) {op} %s ESCAPE '\\\\'")
                params.append('%'+escaped+'%')
        sql = 'SELECT * FROM rets_property'
        if clauses:
            sql += ' WHERE ' + ' AND '.join(clauses)
        if 'sort' in filters:
            ordering = {'price_asc':'L_SystemPrice ASC','price_desc':'L_SystemPrice DESC'}
            if filters['sort'] not in ordering:
                raise ValueError('Invalid sort option')
            sql += ' ORDER BY ' + ordering[filters['sort']]
        return sql, params


class SchemaValidator:
    def __init__(self, schema_path='data/schema.json'):
        self.schema = json.loads(Path(schema_path).read_text(encoding='utf-8'))
        self.cities = {str(c).lower() for c in self.schema.get('valid_cities',[])}
        self.ranges = self.schema.get('ranges',{})

    def validate_query(self, filters):
        errors=[]
        if 'city' in filters:
            if not self.cities:
                errors.append('No city list loaded; city cannot be validated')
            elif str(filters['city']).lower() not in self.cities:
                errors.append(f"City '{filters['city']}' not found in database")
        for root,minimum,maximum in [('price',100000,10000000),('bedrooms',0,10),('bathrooms',0,10)]:
            lo=self.ranges.get(root+'_min',minimum)
            hi=self.ranges.get(root+'_max',maximum)
            for key in (root,root+'_min',root+'_max'):
                if key in filters:
                    value=filters[key]
                    if not isinstance(value,(float,int)) or not lo<=value<=hi:
                        errors.append(f'{key} outside range {lo}–{hi}: {value}')
            if root+'_min' in filters and root+'_max' in filters and filters[root+'_min']>filters[root+'_max']:
                errors.append(f'{root}_min cannot exceed {root}_max')
        if set(filters.get('amenities_include',[])) & set(filters.get('amenities_exclude',[])):
            errors.append('Conflicting include/exclude amenities')
        return not errors, errors


if __name__ == '__main__':
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument('query')
    args=cli.parse_args()
    parser=QueryParser()
    filters=parser.parse(args.query)
    print('Filters:',json.dumps(filters,indent=2))
    if Path('data/schema.json').exists():
        valid,errors=SchemaValidator().validate_query(filters)
        print('Valid:',valid,'Errors:',errors)
        if not valid:
            raise SystemExit(1)
    try:
        sql,params=parser.to_sql(filters)
    except ValueError as exc:
        print('SQL unavailable:',exc)
    else:
        print('SQL:',sql,'\nParameters:',params)
