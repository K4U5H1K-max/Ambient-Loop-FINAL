"""Test script for the ERP simulator integration with the customer support tools."""
import json
from data.service import ERPService
from agent.tools import (
    check_order_status,
    track_order,
    check_stock,
    initialize_resend,
    initialize_refund,
)

def test_order_status():
    """Test the order status tool"""
    print("\n=== Testing Order Status ===")
    # Test with a valid order
    print(check_order_status("ORD12345"))
    
    # Test with another valid order
    print(check_order_status("ORD67890"))
    
    # Test with an invalid order
    print(check_order_status("INVALID123"))

def test_tracking():
    """Test the order tracking tool"""
    print("\n=== Testing Order Tracking ===")
    # Test with a valid order
    print(track_order("ORD12345"))
    
    # Test with an order that has no shipment
    print(track_order("INVALID123"))

def test_stock_check():
    """Test the stock check tool"""
    print("\n=== Testing Stock Check ===")
    # Test with a product that's in stock
    print(check_stock("P1001"))
    
    # Test with a product that's out of stock
    print(check_stock("P1002"))
    
    # Test with another product that's in stock
    print(check_stock("P1003"))
    
    # Test with an invalid product
    print(check_stock("INVALID123"))

def test_resend():
    """Test the resend tool"""
    print("\n=== Testing Resend ===")
    # Test with a valid order and product
    print(initialize_resend("ORD12345/P1001"))
    
    # Test with a valid order but out of stock product
    print(initialize_resend("ORD12345/P1002"))
    
    # Test with an invalid order
    print(initialize_resend("INVALID123/P1001"))

def test_refund():
    """Test the refund tool"""
    print("\n=== Testing Refund ===")
    # Test with a valid order and product
    print(initialize_refund("ORD12345/P1001"))
    
    # Test with an already returned item
    print(initialize_refund("ORD12345/P1001"))
    
    # Test with another valid order and product
    print(initialize_refund("ORD67890/P1002"))
    
    # Test with an invalid order
    print(initialize_refund("INVALID123/P1001"))

def main():
    """Run all tests"""
    print("Starting ERP integration tests...")
    
    test_order_status()
    test_tracking()
    test_stock_check()
    test_resend()
    test_refund()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
