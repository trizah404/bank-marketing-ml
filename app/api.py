from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_bundle, predict_one

app = FastAPI(title="Bank Marketing Predictor")
bundle = load_bundle()

EXAMPLE = {
    "age": 62, "job": "retired", "marital": "married", "education": "university.degree",
    "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
    "month": "oct", "day_of_week": "tue", "campaign": 1, "pdays": 6, "previous": 1,
    "poutcome": "success", "emp.var.rate": -3.4, "cons.price.idx": 92.431,
    "cons.conf.idx": -26.9, "euribor3m": 0.754, "nr.employed": 5017.5,
}


class Customer(BaseModel):
    """What a valid request MUST look like. Anything else -> rejected (error 422)."""
    model_config = ConfigDict(
        extra="forbid",              # unknown fields (e.g. 'duration') are REJECTED
        populate_by_name=True,
        json_schema_extra={"example": EXAMPLE},   # pre-fills the /docs page
    )

    # --- customer profile ---
    age: int = Field(ge=17, le=100)              # ge = greater or equal, le = less or equal
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                 "retired", "self-employed", "services", "student", "technician",
                 "unemployed", "unknown"]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                       "professional.course", "university.degree", "unknown"]
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]

    # --- campaign info (all known BEFORE the call) ---
    contact: Literal["cellular", "telephone"]
    month: Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(ge=1)                  # calls this campaign, incl. this one
    pdays: int = Field(ge=0, le=999)             # 999 = never contacted before
    previous: int = Field(ge=0)
    poutcome: Literal["failure", "nonexistent", "success"]

    # --- economy (dots aren't allowed in Python names, so we use aliases) ---
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")


class Prediction(BaseModel):
    probability: float
    call: bool
    threshold: float
    model: str


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["model_name"], "threshold": bundle["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    # by_alias=True turns emp_var_rate back into "emp.var.rate", the name the model knows
    return predict_one(customer.model_dump(by_alias=True))