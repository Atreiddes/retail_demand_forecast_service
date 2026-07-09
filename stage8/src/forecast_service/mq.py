"""RabbitMQ через pika: публикация пачек и потребление воркером.

Одна durable-очередь с приоритетами: мелкие интерактивные прогоны (один магазин) идут с
высоким приоритетом и не ждут за крупным прогоном по всем магазинам. Публикация пачек
прогона транзакционная: либо в очереди все пачки, либо ни одной (обрыв на середине не
оставляет полупрогон). Воркер потребляет с prefetch=1 и ручным подтверждением.
"""
from __future__ import annotations

import json
import time

import pika

from .config import settings

QUEUE = "forecast_chunks.v3"  # v3: к приоритету добавлен dead-letter для провалившихся пачек
DLQ = "forecast_chunks.dlq"   # провалившаяся пачка уходит сюда, а не теряется - можно разобрать
QUEUE_ARGS = {"x-max-priority": 5,
              "x-dead-letter-exchange": "", "x-dead-letter-routing-key": DLQ}


def connect(attempts=15):
    params = pika.URLParameters(settings.rabbitmq_url)
    for _ in range(attempts):
        try:
            return pika.BlockingConnection(params)
        except pika.exceptions.AMQPConnectionError:
            time.sleep(2)
    raise RuntimeError("RabbitMQ недоступен")


def _declare(ch):
    ch.queue_declare(queue=DLQ, durable=True)
    ch.queue_declare(queue=QUEUE, durable=True, arguments=QUEUE_ARGS)


def publish(messages, priority=1):
    conn = connect(attempts=3)  # в пути HTTP-запроса не висим долго: быстрый отказ -> 503
    try:
        ch = conn.channel()
        _declare(ch)
        ch.tx_select()  # транзакция: пачки прогона попадают в очередь все разом или никак
        for m in messages:
            ch.basic_publish("", QUEUE, json.dumps(m),
                             properties=pika.BasicProperties(delivery_mode=2, priority=priority))
        ch.tx_commit()
    finally:
        conn.close()


def consume(handle):
    conn = connect()
    ch = conn.channel()
    _declare(ch)
    ch.basic_qos(prefetch_count=1)

    def on_msg(c, method, props, body):
        try:
            handle(body)
            c.basic_ack(method.delivery_tag)
        except Exception:
            c.basic_nack(method.delivery_tag, requeue=False)  # dead-letter в DLQ, не теряется

    ch.basic_consume(QUEUE, on_msg)
    ch.start_consuming()
