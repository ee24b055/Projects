import asyncio
from decimal import Decimal
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Numeric, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ---- DATABASE CONFIGURATION ----
# We use SQLite for an instant, zero-setup relational database
DATABASE_URL = "sqlite:///./matching_engine.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---- SQLALCHEMY DB MODELS ----
class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    balance = Column(Numeric(12, 2), default=10000.00) # Simulating starting cash

class DBOrder(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ticker = Column(String, index=True)
    side = Column(String)  # "BUY" or "SELL"
    price = Column(Numeric(10, 2))
    quantity = Column(Integer)
    status = Column(String, default="PENDING")  # PENDING, FILLED

class DBTrade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    buyer_id = Column(Integer, ForeignKey("users.id"))
    seller_id = Column(Integer, ForeignKey("users.id"))
    ticker = Column(String)
    price = Column(Numeric(10, 2))
    quantity = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the tables right now
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# ---- PYDANTIC SCHEMAS (API Validation) ----
class UserCreate(BaseModel):
    username: str

class OrderCreate(BaseModel):
    user_id: int
    ticker: str
    side: str = Field(..., pattern="^(BUY|SELL)$")
    price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)

# ---- IN-MEMORY ORDER BOOK STRUCTURE ----
class OrderBook:
    def __init__(self):
        # Format: {"AAPL": {"BUY": [...], "SELL": [...]}}
        self.books: Dict[str, Dict[str, List[dict]]] = {}
        self.lock = asyncio.Lock()  # Prevents Concurrency Race Conditions

    async def add_and_match(self, order: dict, db: Session) -> List[dict]:
        ticker = order["ticker"]
        side = order["side"]
        
        # Thread-safe operation begins here
        async with self.lock:
            if ticker not in self.books:
                self.books[ticker] = {"BUY": [], "SELL": []}
            
            trades_executed = []
            
            if side == "BUY":
                # Match with cheapest SELL orders first
                sells = self.books[ticker]["SELL"]
                while sells and order["quantity"] > 0:
                    best_sell = sells[0]
                    if order["price"] >= best_sell["price"]:
                        # Match Found! Determine execution volume
                        match_qty = min(order["quantity"], best_sell["quantity"])
                        
                        # Execute the trade logic
                        trade = self._execute_trade_record(order, best_sell, match_qty, db)
                        trades_executed.append(trade)
                        
                        # Update remaining quantities
                        order["quantity"] -= match_qty
                        best_sell["quantity"] -= match_qty
                        
                        if best_sell["quantity"] == 0:
                            sells.pop(0)
                    else:
                        break # Best sell price is too high for our buy limit
                
                # If order is not fully filled, add remaining to BUY book sorted descending
                if order["quantity"] > 0:
                    self.books[ticker]["BUY"].append(order)
                    self.books[ticker]["BUY"].sort(key=lambda x: x["price"], reverse=True)
                    
            else: # side == "SELL"
                # Match with highest BUY orders first
                buys = self.books[ticker]["BUY"]
                while buys and order["quantity"] > 0:
                    best_buy = buys[0]
                    if order["price"] <= best_buy["price"]:
                        match_qty = min(order["quantity"], best_buy["quantity"])
                        
                        trade = self._execute_trade_record(best_buy, order, match_qty, db)
                        trades_executed.append(trade)
                        
                        order["quantity"] -= match_qty
                        best_buy["quantity"] -= match_qty
                        
                        if best_buy["quantity"] == 0:
                            buys.pop(0)
                    else:
                        break
                
                # If order is not fully filled, add remaining to SELL book sorted ascending
                if order["quantity"] > 0:
                    self.books[ticker]["SELL"].append(order)
                    self.books[ticker]["SELL"].sort(key=lambda x: x["price"])
            
            return trades_executed

    def _execute_trade_record(self, buy_order: dict, sell_order: dict, qty: int, db: Session):
        # Standardize execution price to the order already sitting in the book
        exec_price = sell_order["price"] if buy_order["timestamp"] > sell_order["timestamp"] else buy_order["price"]
        
        # Fetch users from DB to update balances (Atomic transaction context)
        buyer = db.query(DBUser).filter(DBUser.id == buy_order["user_id"]).first()
        seller = db.query(DBUser).filter(DBUser.id == sell_order["user_id"]).first()
        
        cost = Decimal(str(exec_price)) * qty
        buyer.balance -= cost
        seller.balance += cost
        
        # Log trade in DB
        db_trade = DBTrade(
            buyer_id=buyer.id, seller_id=seller.id,
            ticker=buy_order["ticker"], price=exec_price, quantity=qty
        )
        db.add(db_trade)
        
        # Update original DB orders status if completely filled
        if buy_order["quantity"] == qty:
            db.query(DBOrder).filter(DBOrder.id == buy_order["db_id"]).update({"status": "FILLED"})
        if sell_order["quantity"] == qty:
            db.query(DBOrder).filter(DBOrder.id == sell_order["db_id"]).update({"status": "FILLED"})
            
        db.commit()
        return {"buyer_id": buyer.id, "seller_id": seller.id, "price": exec_price, "qty": qty}

# Initialize global order book
order_book_engine = OrderBook()


# ---- FASTAPI ROUTER SETUP ----
app = FastAPI(title="High-Performance Matching Engine API")

@app.post("/users/", status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = DBUser(username=user.username)
    db.add(db_user)
    try:
        db.commit()
        db.refresh(db_user)
        return {"user_id": db_user.id, "username": db_user.username, "balance": db_user.balance}
    except Exception:
        db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/orders/")
async def place_order(order: OrderCreate, db: Session = Depends(get_db)):
    # 1. Verify user exists
    user = db.query(DBUser).filter(DBUser.id == order.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # 2. Financial validation: ensure buyer has enough funds
    if order.side == "BUY" and user.balance < (Decimal(str(order.price)) * order.quantity):
        raise HTTPException(status_code=400, detail="Insufficient funds")

    # 3. Log initial pending order to database
    db_order = DBOrder(
        user_id=order.user_id, ticker=order.ticker.upper(),
        side=order.side, price=order.price, quantity=order.quantity
    )
    db.add(db_order)
    db.commit()
    db.refresh(db_order)

    # 4. Construct memory-dict to push to matching engine
    engine_payload = {
        "db_id": db_order.id,
        "user_id": order.user_id,
        "ticker": order.ticker.upper(),
        "side": order.side,
        "price": order.price,
        "quantity": order.quantity,
        "timestamp": datetime.utcnow().timestamp()
    }

    # 5. Send to matching engine engine
    trades = await order_book_engine.add_and_match(engine_payload, db)
    
    return {
        "message": "Order processed",
        "order_id": db_order.id,
        "executed_trades": trades
    }

@app.get("/orderbook/{ticker}")
def get_order_book(ticker: str):
    tk = ticker.upper()
    book = order_book_engine.books.get(tk, {"BUY": [], "SELL": []})
    return {
        "ticker": tk,
        "buy_depth": [{"price": o["price"], "qty": o["quantity"]} for o in book["BUY"]],
        "sell_depth": [{"price": o["price"], "qty": o["quantity"]} for o in book["SELL"]]
    }