import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference.generate import generate


