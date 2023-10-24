from pydantic import BaseModel
from typing import Optional

class InputSchema(BaseModel):
	id: Optional[int] = None
	led_red : str
	led_green : str
	led_mirrow : str


