"""ERP system simulation."""
from erp.models import (
    Product, InventoryItem, Shipment, Order, OrderItem, Customer,
    TrackingEvent, ReturnRequest, OrderStatus, ShipmentStatus, ProductCategory
)
from erp.data import PRODUCTS, INVENTORY, ORDERS, SHIPMENTS, CUSTOMERS, RETURN_REQUESTS
from erp.service import ERPService

__all__ = [
    "Product", "InventoryItem", "Shipment", "Order", "OrderItem", "Customer",
    "TrackingEvent", "ReturnRequest", "OrderStatus", "ShipmentStatus", "ProductCategory",
    "PRODUCTS", "INVENTORY", "ORDERS", "SHIPMENTS", "CUSTOMERS", "RETURN_REQUESTS",
    "ERPService",
]
