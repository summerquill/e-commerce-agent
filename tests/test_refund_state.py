import copy
import unittest

from app.agent.tools.mock_data import ORDERS
from app.agent.tools.order import query_order
from app.agent.tools.refund import apply_refund


class TestRefundState(unittest.TestCase):
    """验证退款会真正写入订单状态，而不只是返回成功文案。"""

    def setUp(self):
        self._snapshot = copy.deepcopy(ORDERS)

    def tearDown(self):
        ORDERS.clear()
        ORDERS.update(copy.deepcopy(self._snapshot))

    def test_pending_order_refund_updates_status(self):
        result = apply_refund("ORD-20240120-002", "不想要了")
        self.assertTrue(result["success"])

        order = query_order("ORD-20240120-002")["order"]
        self.assertEqual(order["status"], "refund_processing")
        self.assertEqual(order["refund_status"], "退款中")
        self.assertEqual(order["refund_reason"], "不想要了")
        self.assertTrue(order["refund_requested_at"])

    def test_shipped_and_delivered_orders_enter_review(self):
        for order_id in ("ORD-20240115-001", "ORD-20240110-003"):
            with self.subTest(order_id=order_id):
                result = apply_refund(order_id, "尺码不合适")
                self.assertTrue(result["success"])

                order = query_order(order_id)["order"]
                self.assertEqual(order["status"], "refund_processing")
                self.assertEqual(order["refund_status"], "审核中")
                self.assertEqual(order["refund_reason"], "尺码不合适")

    def test_duplicate_refund_is_rejected_and_keeps_first_reason(self):
        first = apply_refund("ORD-20240115-001", "尺码不合适")
        self.assertTrue(first["success"])

        second = apply_refund("ORD-20240115-001", "质量问题")
        self.assertFalse(second["success"])

        order = query_order("ORD-20240115-001")["order"]
        self.assertEqual(order["status"], "refund_processing")
        self.assertEqual(order["refund_reason"], "尺码不合适")

    def test_unknown_order_does_not_mutate_data(self):
        before = copy.deepcopy(ORDERS)
        result = apply_refund("ORD-NOT-EXIST", "不想要了")
        self.assertFalse(result["success"])
        self.assertEqual(ORDERS, before)


if __name__ == "__main__":
    unittest.main()
