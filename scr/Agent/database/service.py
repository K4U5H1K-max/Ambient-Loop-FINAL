"""
ERP System Simulator Service Layer
"""
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import uuid
import random

from database.models import (
    Product, InventoryItem, Shipment, Order, OrderItem, Customer,
    TrackingEvent, ReturnRequest, OrderStatus, ShipmentStatus
)
from database.data import PRODUCTS, INVENTORY, ORDERS, SHIPMENTS, CUSTOMERS, RETURN_REQUESTS

class ERPService:
    """Service layer for interacting with the ERP system data"""
    
    @staticmethod
    def get_order(order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details by order ID
        
        Args:
            order_id: The order ID to look up
            
        Returns:
            Order details or None if not found
        """
        if order_id in ORDERS:
            order = ORDERS[order_id]
            # Convert to dict and add product details
            order_dict = {
                "id": order.id,
                "customer_id": order.customer_id,
                "customer_name": CUSTOMERS[order.customer_id].name if order.customer_id in CUSTOMERS else "Unknown",
                "status": order.status.value,
                "items": [],
                "total_amount": order.total_amount,
                "created_at": order.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "shipment_id": order.shipment_id
            }
            
            # Add items with product details
            for item in order.items:
                product = PRODUCTS.get(item.product_id)
                item_dict = {
                    "product_id": item.product_id,
                    "product_name": product.name if product else "Unknown Product",
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_price": item.total_price,
                    "is_returned": item.is_returned,
                    "return_reason": item.return_reason
                }
                order_dict["items"].append(item_dict)
                
            return order_dict
        return None
    
    @staticmethod
    def get_tracking_info(order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tracking information for an order
        
        Args:
            order_id: The order ID to get tracking for
            
        Returns:
            Tracking information or None if not found
        """
        order = ORDERS.get(order_id)
        if not order or not order.shipment_id:
            return None
            
        shipment = SHIPMENTS.get(order.shipment_id)
        if not shipment:
            return None
            
        tracking_info = {
            "carrier": shipment.carrier,
            "tracking_number": shipment.tracking_number,
            "status": shipment.status.value,
            "estimated_delivery": shipment.estimated_delivery.strftime("%Y-%m-%d"),
            "actual_delivery": shipment.actual_delivery.strftime("%Y-%m-%d") if shipment.actual_delivery else None,
            "tracking_history": []
        }
        
        # Add tracking events
        for event in shipment.tracking_history:
            tracking_info["tracking_history"].append({
                "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "location": event.location,
                "status": event.status.value,
                "description": event.description
            })
            
        return tracking_info
    
    @staticmethod
    def check_stock(product_id: str) -> Dict[str, Any]:
        """
        Check stock level for a product
        
        Args:
            product_id: The product ID to check
            
        Returns:
            Stock information
        """
        product = PRODUCTS.get(product_id)
        inventory = INVENTORY.get(product_id)
        
        if not product:
            return {"error": "Product not found", "stock_level": 0, "available": False}
            
        if not inventory:
            return {
                "product_id": product_id,
                "product_name": product.name,
                "stock_level": 0,
                "available": False,
                "warehouse": None,
                "restock_expected": "Unknown"
            }
            
        # Calculate expected restock date if quantity is below threshold
        restock_expected = None
        if inventory.quantity <= inventory.reorder_threshold:
            # Random date in the next 1-14 days
            days_to_restock = random.randint(1, 14)
            restock_expected = (datetime.now() + timedelta(days=days_to_restock)).strftime("%Y-%m-%d")
            
        return {
            "product_id": product_id,
            "product_name": product.name,
            "stock_level": inventory.quantity,
            "available": inventory.quantity > 0,
            "warehouse": inventory.warehouse_id,
            "location": inventory.location,
            "restock_expected": restock_expected
        }
    
    @staticmethod
    def process_return(order_id: str, product_id: str, reason: str) -> Dict[str, Any]:
        """
        Process a return request
        
        Args:
            order_id: The order ID
            product_id: The product ID being returned
            reason: The reason for the return
            
        Returns:
            Return request information
        """
        order = ORDERS.get(order_id)
        if not order:
            return {"error": "Order not found", "success": False}
            
        # Check if the product is in the order
        order_item = None
        for item in order.items:
            if item.product_id == product_id:
                order_item = item
                break
                
        if not order_item:
            return {"error": "Product not found in order", "success": False}
            
        # Check if already returned
        if order_item.is_returned:
            return {"error": "Item already returned", "success": False}
            
        # Create a return request
        return_id = f"RET{uuid.uuid4().hex[:4]}"
        return_request = ReturnRequest(
            id=return_id,
            order_id=order_id,
            items=[order_item],
            reason=reason,
            status="pending",
            refund_amount=order_item.total_price
        )
        
        # Mark item as returned
        order_item.is_returned = True
        order_item.return_reason = reason
        
        # Add to return requests
        RETURN_REQUESTS[return_id] = return_request
        
        return {
            "return_id": return_id,
            "order_id": order_id,
            "product_id": product_id,
            "status": "pending",
            "refund_amount": order_item.total_price,
            "success": True
        }
    
    @staticmethod
    def process_resend(order_id: str, product_id: str) -> Dict[str, Any]:
        """
        Process a resend request
        
        Args:
            order_id: The order ID
            product_id: The product ID to resend
            
        Returns:
            Resend request information
        """
        order = ORDERS.get(order_id)
        if not order:
            return {"error": "Order not found", "success": False}
            
        # Check if the product is in the order
        order_item = None
        for item in order.items:
            if item.product_id == product_id:
                order_item = item
                break
                
        if not order_item:
            return {"error": "Product not found in order", "success": False}
            
        # Check stock availability
        stock_info = ERPService.check_stock(product_id)
        if not stock_info.get("available", False):
            return {
                "error": "Product out of stock",
                "success": False,
                "stock_level": stock_info.get("stock_level", 0),
                "restock_expected": stock_info.get("restock_expected")
            }
            
        # Create a new shipment for the resend
        shipment_id = f"SH{uuid.uuid4().hex[:4]}"
        tracking_number = f"RS{uuid.uuid4().hex[:8].upper()}"
        
        # Estimated delivery in 3-5 days
        delivery_days = random.randint(3, 5)
        estimated_delivery = datetime.now() + timedelta(days=delivery_days)
        
        shipment = Shipment(
            id=shipment_id,
            order_id=order_id,
            carrier="FedEx",
            tracking_number=tracking_number,
            status=ShipmentStatus.PROCESSING,
            estimated_delivery=estimated_delivery,
            tracking_history=[
                TrackingEvent(
                    timestamp=datetime.now(),
                    location="Warehouse",
                    status=ShipmentStatus.PROCESSING,
                    description="Replacement order being processed"
                )
            ]
        )
        
        # Add to shipments
        SHIPMENTS[shipment_id] = shipment
        
        # Update inventory
        inventory = INVENTORY.get(product_id)
        if inventory:
            inventory.quantity -= 1
            inventory.updated_at = datetime.now()
        
        return {
            "shipment_id": shipment_id,
            "order_id": order_id,
            "product_id": product_id,
            "tracking_number": tracking_number,
            "estimated_delivery": estimated_delivery.strftime("%Y-%m-%d"),
            "success": True
        }
