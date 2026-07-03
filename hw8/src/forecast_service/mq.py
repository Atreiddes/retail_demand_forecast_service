"""RabbitMQ через pika: публикация пачек и потребление воркером.

Одна durable-очередь forecast_chunks. Публикация открывает короткое соединение (не делим
канал между потоками api). Воркер потребляет с prefetch=1 и ручным подтверждением.
"""
from __future__ import annotations

import json
import time

import pika

from .config import settings

QUEUE = "forecast_chunks"


def connect():
    params = pika.URLParameters(settings.rabbitmq_url)
    for _ in range(15):
        try:
            return pika.BlockingConnection(params)
        except pika.exceptions.AMQPConnectionError:
            time.sleep(2)
    raise RuntimeError("RabbitMQ недоступен")


def publish(messages):
    conn = connect()
    try:
        ch = conn.channel()
        ch.queue_declare(queue=QUEUE, durable=True)
        for m in messages:
            ch.basic_publish("", QUEUE, json.dumps(m),
                             properties=pika.BasicProperties(delivery_mode=2))
    finally:
        conn.close()


def consume(handle):
    conn = connect()
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_msg(c, method, props, body):
        try:
            handle(body)
            c.basic_ack(method.delivery_tag)
        except Exception:
            c.basic_nack(method.delivery_tag, requeue=False)

    ch.basic_consume(QUEUE, on_msg)
    ch.start_consuming()
