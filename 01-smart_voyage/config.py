# -*- coding: utf-8 -*-
"""
途景汇文旅多智能体一体化系统 - 全局配置
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LLMConfig(BaseModel):
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY", ""))
    model: str = Field(default=os.getenv("LLM_MODEL", "qwen-plus"))
    base_url: str = Field(default=os.getenv("LLM_BASE_URL", ""))


class MySQLConfig(BaseModel):
    host: str = Field(default=os.getenv("MYSQL_HOST", "localhost"))
    port: int = Field(default=int(os.getenv("MYSQL_PORT", "3306")))
    user: str = Field(default=os.getenv("MYSQL_USER", "root"))
    password: str = Field(default=os.getenv("MYSQL_PASSWORD", "root"))
    database: str = Field(default=os.getenv("MYSQL_DATABASE", "songjing_hui"))

    @property
    def dsn(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?charset=utf8mb4"


class MCPServerConfig(BaseModel):
    host: str = Field(default=os.getenv("MCP_SERVER_HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("MCP_SERVER_PORT", "8001")))


class AgentServiceConfig(BaseModel):
    weather_port: int = Field(default=int(os.getenv("WEATHER_AGENT_PORT", "8101")))
    attraction_port: int = Field(default=int(os.getenv("ATTRACTION_AGENT_PORT", "8102")))
    route_port: int = Field(default=int(os.getenv("ROUTE_AGENT_PORT", "8103")))
    gateway_port: int = Field(default=int(os.getenv("GATEWAY_PORT", "8000")))

    @property
    def weather_url(self) -> str:
        return os.getenv("WEATHER_AGENT_URL", f"http://localhost:{self.weather_port}")

    @property
    def attraction_url(self) -> str:
        return os.getenv("ATTRACTION_AGENT_URL", f"http://localhost:{self.attraction_port}")

    @property
    def route_url(self) -> str:
        return os.getenv("ROUTE_AGENT_URL", f"http://localhost:{self.route_port}")


class MemoryConfig(BaseModel):
    memory_dir: str = Field(default=os.getenv("MEMORY_DIR", "./data/checkpoints"))
    max_context_tokens: int = Field(default=int(os.getenv("MAX_CONTEXT_TOKENS", "4000")))
    summarizer_model: str = Field(default=os.getenv("SUMMARIZER_MODEL", "qwen-turbo"))


class LoRAConfig(BaseModel):
    lora_model_path: str = Field(default=os.getenv("LORA_MODEL_PATH", "./models/lora"))
    base_model_name: str = Field(default=os.getenv("BASE_MODEL_NAME", "qwen2.5-7b-instruct"))
    train_data_path: str = Field(default=os.getenv("TRAIN_DATA_PATH", "./data/train"))
    output_dir: str = Field(default=os.getenv("OUTPUT_DIR", "./models/lora-output"))


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    mysql: MySQLConfig = MySQLConfig()
    mcp: MCPServerConfig = MCPServerConfig()
    agents: AgentServiceConfig = AgentServiceConfig()
    memory: MemoryConfig = MemoryConfig()
    lora: LoRAConfig = LoRAConfig()


def get_config() -> AppConfig:
    return AppConfig()