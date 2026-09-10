# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()


# @app.get("/")
# def home():
#     return {"message": "Hello FastAPI"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str = "kk"):
#     return {"item_id": item_id, "q": q}

# @app.post("/items/")
# def create_item(item: dict):
#     return {"item": item}
