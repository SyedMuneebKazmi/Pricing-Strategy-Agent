from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from src.agent.engine import PricingAgent

router = APIRouter()

class PricingRequest(BaseModel):
    product_id: str
    cost_price: float
    competitor_price: Optional[float] = None
    demand_level: Optional[str] = None
    min_margin: Optional[float] = 0.1
    max_price: Optional[float] = None

@router.post("/recommend")
def recommend(req: PricingRequest):
    agent = PricingAgent()
    result = agent.recommend_price(
        product_id=req.product_id,
        cost_price=req.cost_price,
        competitor_price=req.competitor_price,
        demand_level=req.demand_level,
        min_margin=req.min_margin,
        max_price=req.max_price,
    )
    return result
