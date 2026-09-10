# FastAPI 使用指南

FastAPI 是一个现代、高性能的 Python Web 框架，基于 ASGI 标准。

## 安装

使用 pip 安装 FastAPI 和 Uvicorn 服务器：

```
pip install fastapi uvicorn
```

## 快速开始

用装饰器定义路由，用 Pydantic 模型做参数校验，FastAPI 会自动生成交互式文档。

## 路径参数

路径参数是 URL 路径中的一部分，例如 /users/{user_id}。
在函数签名中声明同名参数即可接收，FastAPI 会做类型校验。

## 查询参数

查询参数是 URL 中问号后面的键值对，例如 /items?skip=0&limit=10。
不在路径中声明的函数参数默认会被当作查询参数处理。

## 请求体

请求体通常用 Pydantic 模型定义，FastAPI 会自动校验字段类型、
执行约束并生成 OpenAPI 文档。

## 依赖注入

FastAPI 的依赖注入系统可以复用公共逻辑，例如鉴权、数据库连接、
配置读取等，减少重复代码。

## 异步支持

FastAPI 原生支持 async/await，异步路由可以并发处理请求，
适合 I/O 密集型场景，例如调用外部 LLM API。
