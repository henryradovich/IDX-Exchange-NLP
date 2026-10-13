#!/usr/bin/env python3
import json, re
from pathlib import Path

class QueryParser:
    def __init__(self, taxonomy_path="data/processed/taxonomy.json"):
        self.taxonomy_path = Path(taxonomy_path)
        self.amenity_terms = self._load_terms()

    def _load_terms(self):
        if not self.taxonomy_path.exists():
            return []
        try:
            data = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        keep = {"amenities","property features","exterior","location","interior","rooms","condition"}
        terms = []
        for item in data.get("terms", []):
            term = str(item.get("term","")).strip().lower()
            cat = str(item.get("category","")).strip().lower()
            if term and cat in keep:
                terms.append(term)
        return sorted(set(terms), key=lambda x: (-len(x), x))

    @staticmethod
    def _num(number, suffix=""):
        value = float(str(number).replace(",",""))
        if suffix.lower() == "k":
            value *= 1000
        elif suffix.lower() == "m":
            value *= 1000000
        return int(value)

    def parse(self, query):
        q = query.strip()
        s = q.lower()
        f = {}

        for pat, key in [
            (r"\b(?:under|below|less than)\s+\$?([\d,.]+)\s*([km]?)\b","price_max"),
            (r"\b(?:over|above|more than)\s+\$?([\d,.]+)\s*([km]?)\b","price_min"),
        ]:
            m = re.search(pat, s, re.I)
            if m:
                f[key] = self._num(m.group(1), m.group(2))

        m = re.search(r"\bbetween\s+\$?([\d,.]+)\s*([km]?)\s+(?:and|to|-)\s+\$?([\d,.]+)\s*([km]?)\b", s, re.I)
        if m:
            f["price_min"] = self._num(m.group(1), m.group(2))
            f["price_max"] = self._num(m.group(3), m.group(4))

        m = re.search(r"\b\$?([\d,.]+)\s*([km]?)\s*(?:-|to)\s*\$?([\d,.]+)\s*([km]?)\b", s, re.I)
        if m and "price_min" not in f and "price_max" not in f:
            f["price_min"] = self._num(m.group(1), m.group(2))
            f["price_max"] = self._num(m.group(3), m.group(4))

        m = re.search(r"\b(\d+)\s*\+\s*(?:bed|beds|bedroom|bedrooms|br|bd)\b", s, re.I)
        if m:
            f["bedrooms_min"] = int(m.group(1))
        else:
            m = re.search(r"\bat least\s+(\d+)\s*(?:bed|beds|bedroom|bedrooms|br|bd)\b", s, re.I)
            if m:
                f["bedrooms_min"] = int(m.group(1))
            else:
                m = re.search(r"\b(?:up to|no more than)\s+(\d+)\s*(?:bed|beds|bedroom|bedrooms|br|bd)\b", s, re.I)
                if m:
                    f["bedrooms_max"] = int(m.group(1))
                else:
                    m = re.search(r"\b(\d+)\s*(?:bed|beds|bedroom|bedrooms|br|bd)\b", s, re.I)
                    if m:
                        f["bedrooms"] = int(m.group(1))

        m = re.search(r"\b(\d+(?:\.\d+)?)\s*\+\s*(?:bath|baths|bathroom|bathrooms|ba)\b", s, re.I)
        if m:
            f["bathrooms_min"] = float(m.group(1))
        else:
            m = re.search(r"\bat least\s+(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", s, re.I)
            if m:
                f["bathrooms_min"] = float(m.group(1))
            else:
                m = re.search(r"\b(?:up to|no more than)\s+(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", s, re.I)
                if m:
                    f["bathrooms_max"] = float(m.group(1))
                else:
                    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", s, re.I)
                    if m:
                        f["bathrooms"] = float(m.group(1))

        m = re.search(r"\bin\s+([A-Za-z .'-]+?)(?=\s+(?:under|below|over|above|with|without|at least|up to|between)\b|$)", q, re.I)
        if m:
            city = m.group(1).strip(" ,.")
            if city:
                f["city"] = city

        statuses = {
            "active": r"\bactive listings?\b",
            "pending": r"\bpending\b",
            "under_contract": r"\bunder contract\b",
            "contingent": r"\bcontingent\b",
            "sold": r"\b(?:recently )?sold\b",
            "new_listing": r"\bnew listings?\b",
            "back_on_market": r"\bback on (?:the )?market\b",
        }
        for value, pat in statuses.items():
            if re.search(pat, s):
                f["listing_status"] = value
                break

        if re.search(r"\bno hoa\b|\bwithout hoa\b", s):
            f["hoa"] = False

        inc, exc = [], []
        for term in self.amenity_terms:
            pat = re.compile(r"(?<!\w)" + re.escape(term).replace(r"\ ", r"[\s-]+") + r"(?!\w)", re.I)
            m = pat.search(s)
            if not m:
                continue
            context = s[max(0,m.start()-25):m.start()]
            if re.search(r"\b(?:no|without|exclude|excluding|not)\s*$", context):
                exc.append(term)
            else:
                inc.append(term)
        if inc:
            f["amenities_include"] = sorted(set(inc))
        if exc:
            f["amenities_exclude"] = sorted(set(exc))
        return f

    def to_sql(self, filters):
        conditions, params = [], []
        def add(col, op, val):
            conditions.append(f"{col} {op} %s")
            params.append(val)

        for key,col,op in [
            ("price_min","L_SystemPrice",">="),("price_max","L_SystemPrice","<="),
            ("bedrooms","L_Keyword2","="),("bedrooms_min","L_Keyword2",">="),("bedrooms_max","L_Keyword2","<="),
            ("bathrooms","LM_Dec_3","="),("bathrooms_min","LM_Dec_3",">="),("bathrooms_max","LM_Dec_3","<="),
            ("city","L_City","="),
        ]:
            if key in filters:
                add(col,op,filters[key])

        for term in filters.get("amenities_include",[]):
            conditions.append("LOWER(L_Remarks) LIKE %s")
            params.append(f"%{term.lower()}%")
        for term in filters.get("amenities_exclude",[]):
            conditions.append("LOWER(L_Remarks) NOT LIKE %s")
            params.append(f"%{term.lower()}%")

        sql = "SELECT * FROM rets_property"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return sql, params


class SchemaValidator:
    def __init__(self, schema_path="data/processed/schema.json"):
        self.schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        self.valid_cities = {c.lower(): c for c in self.schema.get("valid_cities",[])}
        self.ranges = self.schema.get("ranges",{})

    def validate_query(self, filters):
        errors = []
        if "city" in filters and self.valid_cities:
            if str(filters["city"]).lower() not in self.valid_cities:
                errors.append(f"City '{filters['city']}' not found in database")

        low, high = self.ranges.get("price_min",100000), self.ranges.get("price_max",10000000)
        for key in ("price_min","price_max"):
            if key in filters and not (low <= filters[key] <= high):
                errors.append(f"{key} value {filters[key]} outside typical range {low}-{high}")
        if "price_min" in filters and "price_max" in filters and filters["price_min"] > filters["price_max"]:
            errors.append("price_min cannot exceed price_max")

        for key in ("bedrooms","bedrooms_min","bedrooms_max"):
            if key in filters and not (0 <= filters[key] <= self.ranges.get("bedrooms_max",10)):
                errors.append(f"Bedroom count {filters[key]} seems invalid")
        for key in ("bathrooms","bathrooms_min","bathrooms_max"):
            if key in filters and not (0 <= filters[key] <= self.ranges.get("bathrooms_max",10)):
                errors.append(f"Bathroom count {filters[key]} seems invalid")

        return len(errors) == 0, errors
