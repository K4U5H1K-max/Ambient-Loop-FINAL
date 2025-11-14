from database.ticket_db import Product
from database.models import ProductCategory  # use enum from models.py
from database.ticket_db import get_session
from database.data import PRODUCTS

def seed_products():
    """Populate the products table using entries from data.py."""
    session = get_session()
    print("🌱 Seeding products...")

    for product_id, product_data in PRODUCTS.items():
        existing = session.query(Product).filter_by(id=product_id).first()

        # Step 3: Validate category and convert to string
        if isinstance(product_data.category, ProductCategory):
            category_str = product_data.category.value  # always pass string
        elif isinstance(product_data.category, str):
            try:
                category_enum = ProductCategory(product_data.category.lower())
            except ValueError:
                raise ValueError(f"Invalid category for product {product_data.id}: {product_data.category}")
            category_str = category_enum.value
        else:
            raise ValueError(f"Invalid category for product {product_data.id}")

        if not existing:
            session.add(Product(
                id=product_data.id,
                name=product_data.name,
                description=product_data.description,
                price=product_data.price,
                category=category_str,  # pass string to DB
                weight=product_data.weight,
                dimensions=product_data.dimensions
            ))

    session.commit()
    session.close()
    print("✅ Products seeded successfully.")

if __name__ == "__main__":
    seed_products()
