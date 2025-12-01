from data.service import ERPService
import json

def check_order_status(order_id: str) -> str:
    """Check the status of an order by its order ID"""
    order_info = ERPService.get_order(order_id)
    
    if not order_info:
        return f"Order {order_id} not found in the system."
    
    # Format a detailed response
    response = f"Order {order_id} Status: {order_info['status'].upper()}\n"
    response += f"Ordered on: {order_info['created_at']}\n"
    response += f"Customer: {order_info['customer_name']}\n\n"
    
    response += "Items:\n"
    for item in order_info['items']:
        return_status = " (RETURNED)" if item['is_returned'] else ""
        response += f"- {item['product_name']} x{item['quantity']} - ${item['total_price']:.2f}{return_status}\n"
    
    response += f"\nTotal amount: ${order_info['total_amount']:.2f}"
    
    return response

def track_order(order_id: str) -> str:
    """Track an order's shipment status by order ID"""
    tracking_info = ERPService.get_tracking_info(order_id)
    
    if not tracking_info:
        return f"No tracking information found for order {order_id}."
    
    # Format a detailed response
    response = f"Tracking for Order {order_id}:\n"
    response += f"Carrier: {tracking_info['carrier']}\n"
    response += f"Tracking Number: {tracking_info['tracking_number']}\n"
    response += f"Current Status: {tracking_info['status'].upper()}\n"
    response += f"Estimated Delivery: {tracking_info['estimated_delivery']}\n"
    
    if tracking_info['actual_delivery']:
        response += f"Delivered on: {tracking_info['actual_delivery']}\n"
    
    response += "\nTracking History:\n"
    for event in tracking_info['tracking_history']:
        response += f"- {event['timestamp']}: {event['status']} - {event['description']} ({event['location']})\n"
    
    return response

def check_stock(product_id: str) -> str:
    """Check the stock level of a product by its product ID or name"""
    from data.data import PRODUCTS
    
    # If input looks like a product ID (P followed by digits), use it directly
    if product_id.startswith('P') and len(product_id) == 5 and product_id[1:].isdigit():
        actual_id = product_id
    else:
        # Try to find product by name
        actual_id = None
        for pid, product in PRODUCTS.items():
            if product.name.lower() == product_id.lower():
                actual_id = pid
                break
        if not actual_id:
            return f"Error: Product not found"
    
    stock_info = ERPService.check_stock(actual_id)
    
    if 'error' in stock_info:
        return f"Error: {stock_info['error']}"
    
    # Format a detailed response
    response = f"Product: {stock_info['product_name']} (ID: {stock_info['product_id']})\n"
    response += f"Stock Level: {stock_info['stock_level']} units\n"
    response += f"Status: {'IN STOCK' if stock_info['available'] else 'OUT OF STOCK'}\n"
    
    if stock_info['warehouse']:
        response += f"Warehouse: {stock_info['warehouse']}\n"
        response += f"Location: {stock_info['location']}\n"
    
    if not stock_info['available'] and stock_info['restock_expected']:
        response += f"Restock Expected: {stock_info['restock_expected']}"
    
    return response

def initialize_resend(order_id: str) -> str:
    """Initialize a resend for a specific order"""
    # Check if there are multiple items to resend (comma-separated)
    if ',' in order_id:
        # Split by comma and process each item
        items = order_id.split(',')
        responses = []
        
        for item in items:
            item = item.strip()  # Remove any whitespace
            responses.append(initialize_resend(item))  # Process each item individually
        
        # Combine all responses
        return "\n\n".join(responses)
    
    # Extract product ID from the order_id string if it contains both
    if '/' in order_id:
        order_id, product_id = order_id.split('/')
    else:
        # Get the order details to find the first product
        order_info = ERPService.get_order(order_id)
        if not order_info or not order_info['items']:
            return f"Error: Cannot process resend for order {order_id}. Order not found or has no items."
        product_id = order_info['items'][0]['product_id']
    
    # Process the resend
    result = ERPService.process_resend(order_id, product_id)
    
    if not result['success']:
        if 'stock_level' in result:
            return f"Error: Cannot resend product. {result['error']}. Current stock: {result['stock_level']}. Restock expected: {result.get('restock_expected', 'unknown')}"
        return f"Error: {result['error']}"
    
    # Format a successful response
    response = f"Resend initiated for order {order_id}, product {product_id}\n"
    response += f"New shipment ID: {result['shipment_id']}\n"
    response += f"Tracking number: {result['tracking_number']}\n"
    response += f"Estimated delivery: {result['estimated_delivery']}"
    
    return response

def initialize_refund(order_id: str) -> str:
    """Initialize a refund for a specific order"""
    # Check if there are multiple items to refund (comma-separated)
    if ',' in order_id:
        # Split by comma and process each item
        items = order_id.split(',')
        responses = []
        
        for item in items:
            item = item.strip()  # Remove any whitespace
            responses.append(initialize_refund(item))  # Process each item individually
        
        # Combine all responses
        return "\n\n".join(responses)
    
    # Extract product ID from the order_id string if it contains both
    if '/' in order_id:
        order_id, product_id = order_id.split('/')
    else:
        # Get the order details to find the first product
        order_info = ERPService.get_order(order_id)
        if not order_info or not order_info['items']:
            return f"Error: Cannot process refund for order {order_id}. Order not found or has no items."
        product_id = order_info['items'][0]['product_id']
    
    # Process the return/refund
    result = ERPService.process_return(order_id, product_id, "Customer service approved refund")
    
    if not result['success']:
        return f"Error: {result['error']}"
    
    # Format a successful response
    response = f"Refund initiated for order {order_id}, product {product_id}\n"
    response += f"Return ID: {result['return_id']}\n"
    response += f"Status: {result['status']}\n"
    response += f"Refund amount: ${result['refund_amount']:.2f}"
    
    return response
