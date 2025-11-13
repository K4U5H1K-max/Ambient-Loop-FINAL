"""
ERP System Simulator Sample Data
"""
from datetime import datetime, timedelta
import random
from typing import Dict, List
import uuid

from database.models import (
    Product, InventoryItem, Shipment, Order, OrderItem, Customer,
    TrackingEvent, ReturnRequest, OrderStatus, ShipmentStatus, ProductCategory
)

# Current time reference
NOW = datetime.now()

# Sample Products
PRODUCTS = {
    "P1001": Product(
        id="P1001",
        name="Premium Wireless Headphones",
        description="Noise-cancelling over-ear wireless headphones with 30-hour battery life",
        price=199.99,
        category=ProductCategory.ELECTRONICS,
        weight=0.25,
        dimensions={"length": 18, "width": 15, "height": 8}
    ),
    "P1002": Product(
        id="P1002",
        name="Smart Fitness Watch",
        description="Waterproof fitness tracker with heart rate monitor and GPS",
        price=149.99,
        category=ProductCategory.ELECTRONICS,
        weight=0.05,
        dimensions={"length": 4.5, "width": 3.8, "height": 1.2}
    ),
    "P1003": Product(
        id="P1003",
        name="Organic Cotton T-Shirt",
        description="Soft, breathable 100% organic cotton t-shirt",
        price=29.99,
        category=ProductCategory.CLOTHING,
        weight=0.15,
        dimensions={"length": 28, "width": 20, "height": 2}
    ),
    "P1004": Product(
        id="P1004",
        name="Stainless Steel Water Bottle",
        description="Vacuum insulated water bottle that keeps drinks cold for 24 hours",
        price=34.99,
        category=ProductCategory.HOME,
        weight=0.35,
        dimensions={"length": 27, "width": 7, "height": 7}
    ),
    "P1005": Product(
        id="P1005",
        name="Wireless Charging Pad",
        description="Fast-charging wireless charger compatible with all Qi-enabled devices",
        price=39.99,
        category=ProductCategory.ELECTRONICS,
        weight=0.1,
        dimensions={"length": 10, "width": 10, "height": 1}
    ),
}

# Sample Inventory
INVENTORY = {
    "P1001": InventoryItem(
        product_id="P1001",
        quantity=45,
        warehouse_id="W001",
        location="A12-B3",
        last_restock_date=NOW - timedelta(days=15),
        reorder_threshold=10,
        reorder_quantity=50
    ),
    "P1002": InventoryItem(
        product_id="P1002",
        quantity=0,  # Out of stock
        warehouse_id="W001",
        location="A14-C2",
        last_restock_date=NOW - timedelta(days=30),
        reorder_threshold=5,
        reorder_quantity=25
    ),
    "P1003": InventoryItem(
        product_id="P1003",
        quantity=120,
        warehouse_id="W002",
        location="B22-D5",
        last_restock_date=NOW - timedelta(days=7),
        reorder_threshold=20,
        reorder_quantity=100
    ),
    "P1004": InventoryItem(
        product_id="P1004",
        quantity=78,
        warehouse_id="W001",
        location="C05-A1",
        last_restock_date=NOW - timedelta(days=21),
        reorder_threshold=15,
        reorder_quantity=50
    ),
    "P1005": InventoryItem(
        product_id="P1005",
        quantity=3,  # Low stock
        warehouse_id="W001",
        location="A10-D4",
        last_restock_date=NOW - timedelta(days=25),
        reorder_threshold=10,
        reorder_quantity=30
    ),
}

# Sample Customers
CUSTOMERS = {
    "C1001": Customer(
        id="C1001",
        name="John Smith",
        email="john.smith@example.com",
        phone="555-123-4567",
        address={
            "street": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip": "62704",
            "country": "USA"
        }
    ),
    "C1002": Customer(
        id="C1002",
        name="Emily Johnson",
        email="emily.johnson@example.com",
        phone="555-987-6543",
        address={
            "street": "456 Oak Ave",
            "city": "Riverside",
            "state": "CA",
            "zip": "92501",
            "country": "USA"
        }
    ),
    "C1003": Customer(
        id="C1003",
        name="Michael Brown",
        email="michael.brown@example.com",
        phone="555-456-7890",
        address={
            "street": "789 Pine Rd",
            "city": "Portland",
            "state": "OR",
            "zip": "97201",
            "country": "USA"
        }
    ),
}

# Generate tracking history for a shipment
def generate_tracking_history(days_ago_shipped: int, status: ShipmentStatus) -> List[TrackingEvent]:
    history = []
    
    # Processing at warehouse
    start_date = NOW - timedelta(days=days_ago_shipped)
    history.append(TrackingEvent(
        timestamp=start_date,
        location="Warehouse #1, Springfield, IL",
        status=ShipmentStatus.PROCESSING,
        description="Package processed at shipping facility"
    ))
    
    # In transit
    if days_ago_shipped > 1:
        history.append(TrackingEvent(
            timestamp=start_date + timedelta(hours=12),
            location="Springfield Distribution Center, IL",
            status=ShipmentStatus.IN_TRANSIT,
            description="Package in transit to next facility"
        ))
    
    # More transit events for longer shipments
    if days_ago_shipped > 2:
        history.append(TrackingEvent(
            timestamp=start_date + timedelta(days=1),
            location="Chicago Sorting Center, IL",
            status=ShipmentStatus.IN_TRANSIT,
            description="Package arrived at sorting facility"
        ))
        
    if days_ago_shipped > 3:
        history.append(TrackingEvent(
            timestamp=start_date + timedelta(days=1, hours=12),
            location="Regional Distribution Center",
            status=ShipmentStatus.IN_TRANSIT,
            description="Package in transit to destination"
        ))
    
    # Out for delivery or delivered
    if status in [ShipmentStatus.DELIVERED, ShipmentStatus.OUT_FOR_DELIVERY]:
        if days_ago_shipped > 1:
            history.append(TrackingEvent(
                timestamp=NOW - timedelta(hours=8) if status == ShipmentStatus.OUT_FOR_DELIVERY else NOW - timedelta(days=1),
                location="Local Delivery Facility",
                status=ShipmentStatus.OUT_FOR_DELIVERY,
                description="Package out for delivery"
            ))
        
        if status == ShipmentStatus.DELIVERED:
            history.append(TrackingEvent(
                timestamp=NOW - timedelta(hours=random.randint(1, 6)),
                location="Destination",
                status=ShipmentStatus.DELIVERED,
                description="Package delivered"
            ))
    
    # Failed delivery attempt
    elif status == ShipmentStatus.FAILED:
        history.append(TrackingEvent(
            timestamp=NOW - timedelta(days=1),
            location="Destination",
            status=ShipmentStatus.FAILED,
            description="Delivery attempt failed: No one available to receive package"
        ))
    
    return history

# Sample Shipments
SHIPMENTS = {
    "SH1001": Shipment(
        id="SH1001",
        order_id="ORD12345",
        carrier="FedEx",
        tracking_number="FDX123456789",
        status=ShipmentStatus.DELIVERED,
        estimated_delivery=NOW - timedelta(days=1),
        actual_delivery=NOW - timedelta(hours=30),
        tracking_history=generate_tracking_history(5, ShipmentStatus.DELIVERED),
        created_at=NOW - timedelta(days=6)
    ),
    "SH1002": Shipment(
        id="SH1002",
        order_id="ORD67890",
        carrier="UPS",
        tracking_number="UPS987654321",
        status=ShipmentStatus.DELIVERED,
        estimated_delivery=NOW - timedelta(days=2),
        actual_delivery=NOW - timedelta(hours=50),
        tracking_history=generate_tracking_history(4, ShipmentStatus.DELIVERED),
        created_at=NOW - timedelta(days=5)
    ),
    "SH1003": Shipment(
        id="SH1003",
        order_id="ORD54321",
        carrier="USPS",
        tracking_number="USPS567891234",
        status=ShipmentStatus.IN_TRANSIT,
        estimated_delivery=NOW + timedelta(days=1),
        tracking_history=generate_tracking_history(2, ShipmentStatus.IN_TRANSIT),
        created_at=NOW - timedelta(days=3)
    ),
    "SH1004": Shipment(
        id="SH1004",
        order_id="ORD13579",
        carrier="DHL",
        tracking_number="DHL246813579",
        status=ShipmentStatus.OUT_FOR_DELIVERY,
        estimated_delivery=NOW,
        tracking_history=generate_tracking_history(3, ShipmentStatus.OUT_FOR_DELIVERY),
        created_at=NOW - timedelta(days=4)
    ),
}

# Sample Orders
ORDERS = {
    "ORD12345": Order(
        id="ORD12345",
        customer_id="C1001",
        status=OrderStatus.DELIVERED,
        items=[
            OrderItem(
                product_id="P1001",
                quantity=1,
                unit_price=PRODUCTS["P1001"].price,
                total_price=PRODUCTS["P1001"].price,
                is_returned=False
            ),
            OrderItem(
                product_id="P1004",
                quantity=2,
                unit_price=PRODUCTS["P1004"].price,
                total_price=PRODUCTS["P1004"].price * 2,
                is_returned=False
            )
        ],
        total_amount=PRODUCTS["P1001"].price + (PRODUCTS["P1004"].price * 2),
        shipping_address=CUSTOMERS["C1001"].address,
        billing_address=CUSTOMERS["C1001"].address,
        payment_method="Credit Card",
        shipment_id="SH1001",
        created_at=NOW - timedelta(days=6)
    ),
    "ORD67890": Order(
        id="ORD67890",
        customer_id="C1002",
        status=OrderStatus.DELIVERED,
        items=[
            OrderItem(
                product_id="P1002",
                quantity=1,
                unit_price=PRODUCTS["P1002"].price,
                total_price=PRODUCTS["P1002"].price,
                is_returned=False
            )
        ],
        total_amount=PRODUCTS["P1002"].price,
        shipping_address=CUSTOMERS["C1002"].address,
        billing_address=CUSTOMERS["C1002"].address,
        payment_method="PayPal",
        shipment_id="SH1002",
        created_at=NOW - timedelta(days=5)
    ),
    "ORD54321": Order(
        id="ORD54321",
        customer_id="C1003",
        status=OrderStatus.SHIPPED,
        items=[
            OrderItem(
                product_id="P1003",
                quantity=3,
                unit_price=PRODUCTS["P1003"].price,
                total_price=PRODUCTS["P1003"].price * 3,
                is_returned=False
            ),
            OrderItem(
                product_id="P1005",
                quantity=1,
                unit_price=PRODUCTS["P1005"].price,
                total_price=PRODUCTS["P1005"].price,
                is_returned=False
            )
        ],
        total_amount=(PRODUCTS["P1003"].price * 3) + PRODUCTS["P1005"].price,
        shipping_address=CUSTOMERS["C1003"].address,
        billing_address=CUSTOMERS["C1003"].address,
        payment_method="Credit Card",
        shipment_id="SH1003",
        created_at=NOW - timedelta(days=3)
    ),
    "ORD13579": Order(
        id="ORD13579",
        customer_id="C1001",
        status=OrderStatus.SHIPPED,
        items=[
            OrderItem(
                product_id="P1005",
                quantity=2,
                unit_price=PRODUCTS["P1005"].price,
                total_price=PRODUCTS["P1005"].price * 2,
                is_returned=False
            )
        ],
        total_amount=PRODUCTS["P1005"].price * 2,
        shipping_address=CUSTOMERS["C1001"].address,
        billing_address=CUSTOMERS["C1001"].address,
        payment_method="Credit Card",
        shipment_id="SH1004",
        created_at=NOW - timedelta(days=4)
    ),
}

# Sample Return Requests
RETURN_REQUESTS = {
    "RET1001": ReturnRequest(
        id="RET1001",
        order_id="ORD12345",
        items=[
            OrderItem(
                product_id="P1001",
                quantity=1,
                unit_price=PRODUCTS["P1001"].price,
                total_price=PRODUCTS["P1001"].price,
                is_returned=True,
                return_reason="Defective product"
            )
        ],
        reason="Headphones not working properly",
        status="approved",
        refund_amount=PRODUCTS["P1001"].price,
        created_at=NOW - timedelta(days=1)
    ),
}
