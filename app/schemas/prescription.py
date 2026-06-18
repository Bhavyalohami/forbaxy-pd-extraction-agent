from typing import Literal

from pydantic import BaseModel, Field

MedicationStatus = Literal["normal_medicated", "under_medicated", "over_medicated", "unclear"]
RiskCategory = Literal["low", "medium", "high", "critical", "unclear"]


class Vitals(BaseModel):
    bp: str = ""
    pulse: str = ""
    spo2: str = ""
    temperature: str = ""
    weight: str = ""
    height: str = ""
    pain_score: str = ""
    rbs: str = ""


class Investigation(BaseModel):
    name: str = ""
    notes: str = ""
    status: str = "ordered"


class Medicine(BaseModel):
    name: str = ""
    dose: str = ""
    frequency: str = ""
    duration: str = ""
    route: str = ""
    instructions: str = ""
    confidence: float = Field(default=0, ge=0, le=1)


class FollowUp(BaseModel):
    date: str = ""
    instruction: str = ""


class AdmissionAdvice(BaseModel):
    advised: bool = False
    reason: str = ""
    ipd_probability: float = Field(default=0, ge=0, le=1)
    risk_category: RiskCategory = "low"


class Consultant(BaseModel):
    name: str = ""
    department: str = ""
    specialty: str = ""


class MedicationAssessment(BaseModel):
    status: MedicationStatus = "unclear"
    confidence: float = Field(default=0, ge=0, le=1)
    rationale: str = ""
    flags: list[str] = Field(default_factory=list)
    missing_treatment_concerns: list[str] = Field(default_factory=list)
    excess_treatment_concerns: list[str] = Field(default_factory=list)
    review_recommended: bool = True


class PDExtraction(BaseModel):
    chief_complaints: list[str] = Field(default_factory=list)
    history: str = ""
    allergies: str = ""
    examination: str = ""
    diagnosis: str = ""
    vitals: Vitals = Field(default_factory=Vitals)
    investigations: list[Investigation] = Field(default_factory=list)
    medicines: list[Medicine] = Field(default_factory=list)
    preventive_advice: str = ""
    follow_up: FollowUp = Field(default_factory=FollowUp)
    admission: AdmissionAdvice = Field(default_factory=AdmissionAdvice)
    consultant: Consultant = Field(default_factory=Consultant)
    medication_assessment: MedicationAssessment = Field(default_factory=MedicationAssessment)
    extraction_confidence: float = Field(default=0, ge=0, le=1)
    unclear_fields: list[str] = Field(default_factory=list)
    notes: str = ""


class PDExtractionResponse(BaseModel):
    pd_extraction: PDExtraction = Field(default_factory=PDExtraction)
