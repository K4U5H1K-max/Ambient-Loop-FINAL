"""
ERP System Simulator Data Models
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional
import uuid
import random

# Current time reference
NOW = datetime.now()

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    RETURNED = "returned"
    CANCELLED = "cancelled"

class ShipmentStatus(Enum):
    PROCESSING = "processing"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETURNED = "returned"

class ProductCategory(Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    HOME = "home"
    TOYS = "toys"
    BOOKS = "books"
    BEAUTY = "beauty"

@dataclass
class Product:
    id: str
    name: str
    description: str
    price: float
    category: ProductCategory
    weight: float  # in kg
    dimensions: Dict[str, float]  # length, width, height in cm
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class InventoryItem:
    product_id: str
    quantity: int
    warehouse_id: str
    location: str  # Shelf/bin location
    last_restock_date: datetime
    reorder_threshold: int
    reorder_quantity: int
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class TrackingEvent:
    timestamp: datetime
    location: str
    status: ShipmentStatus
    description: str

@dataclass
class Shipment:
    id: str
    order_id: str
    carrier: str
    tracking_number: str
    status: ShipmentStatus
    estimated_delivery: datetime
    actual_delivery: Optional[datetime] = None
    tracking_history: List[TrackingEvent] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class OrderItem:
    product_id: str
    quantity: int
    unit_price: float
    total_price: float
    is_returned: bool = False
    return_reason: Optional[str] = None

@dataclass
class Order:
    id: str
    customer_id: str
    status: OrderStatus
    items: List[OrderItem]
    total_amount: float
    shipping_address: Dict[str, str]
    billing_address: Dict[str, str]
    payment_method: str
    shipment_id: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Customer:
    id: str
    name: str
    email: str
    phone: str
    address: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ReturnRequest:
    id: str
    order_id: str
    items: List[OrderItem]
    reason: str
    status: str  # pending, approved, rejected, completed
    refund_amount: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
