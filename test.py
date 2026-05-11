from ml_service import predict_delay


# ══════════════════════════════════════════════════════
# PRINT HELPER
# ══════════════════════════════════════════════════════
def print_result(title, history):
    print("\n" + "=" * 70)
    print(f"📦 {title}")
    print("=" * 70)
    print(predict_delay(history))


# ══════════════════════════════════════════════════════
# 🟢 1. PERFECT / VERY RELIABLE SUPPLIER (RULE ENGINE)
# ══════════════════════════════════════════════════════
history_perfect = [
    {"order_date":"2026-01-01","expected_delivery_date":"2026-01-06","actual_delivery_date":"2026-01-06"},
    {"order_date":"2026-02-01","expected_delivery_date":"2026-02-06","actual_delivery_date":"2026-02-05"},
    {"order_date":"2026-03-01","expected_delivery_date":"2026-03-06","actual_delivery_date":"2026-03-06"},
    {"order_date":"2026-04-01","expected_delivery_date":"2026-04-06","actual_delivery_date":"2026-04-05"},
    {"order_date":"2026-05-01","expected_delivery_date":"2026-05-06","actual_delivery_date":"2026-05-06"},
]


# ══════════════════════════════════════════════════════
# 🟡 2. NORMAL SUPPLIER (MIXED BEHAVIOR → STAT MODEL)
# ══════════════════════════════════════════════════════
history_normal = [
    {"order_date":"2026-01-01","expected_delivery_date":"2026-01-06","actual_delivery_date":"2026-01-07"},
    {"order_date":"2026-02-01","expected_delivery_date":"2026-02-06","actual_delivery_date":"2026-02-06"},
    {"order_date":"2026-03-01","expected_delivery_date":"2026-03-06","actual_delivery_date":"2026-03-08"},
    {"order_date":"2026-04-01","expected_delivery_date":"2026-04-06","actual_delivery_date":"2026-04-05"},
    {"order_date":"2026-05-01","expected_delivery_date":"2026-05-06","actual_delivery_date":"2026-05-07"},
]


# ══════════════════════════════════════════════════════
# 🔴 3. LATE SUPPLIER (ML MODEL)
# ══════════════════════════════════════════════════════
history_late = [
    {"order_date":"2026-01-01","expected_delivery_date":"2026-01-06","actual_delivery_date":"2026-01-12"},
    {"order_date":"2026-02-01","expected_delivery_date":"2026-02-06","actual_delivery_date":"2026-02-13"},
    {"order_date":"2026-03-01","expected_delivery_date":"2026-03-06","actual_delivery_date":"2026-03-15"},
    {"order_date":"2026-04-01","expected_delivery_date":"2026-04-06","actual_delivery_date":"2026-04-10"},
    {"order_date":"2026-05-01","expected_delivery_date":"2026-05-06","actual_delivery_date":"2026-05-14"},
]


# ══════════════════════════════════════════════════════
# 🔥 4. CHAOTIC SUPPLIER (HIGH VARIANCE → ML)
# ══════════════════════════════════════════════════════
history_chaotic = [
    {"order_date":"2026-01-01","expected_delivery_date":"2026-01-06","actual_delivery_date":"2026-01-04"},
    {"order_date":"2026-02-01","expected_delivery_date":"2026-02-06","actual_delivery_date":"2026-02-15"},
    {"order_date":"2026-03-01","expected_delivery_date":"2026-03-06","actual_delivery_date":"2026-03-05"},
    {"order_date":"2026-04-01","expected_delivery_date":"2026-04-06","actual_delivery_date":"2026-04-16"},
    {"order_date":"2026-05-01","expected_delivery_date":"2026-05-06","actual_delivery_date":"2026-05-06"},
]


# ══════════════════════════════════════════════════════
# ⚠️ 5. EXTREME LATE SUPPLIER (VERY BAD → ML)
# ══════════════════════════════════════════════════════
history_extreme_late = [
    {"order_date":"2026-01-01","expected_delivery_date":"2026-01-05","actual_delivery_date":"2026-01-15"},
    {"order_date":"2026-02-01","expected_delivery_date":"2026-02-05","actual_delivery_date":"2026-02-18"},
    {"order_date":"2026-03-01","expected_delivery_date":"2026-03-05","actual_delivery_date":"2026-03-20"},
    {"order_date":"2026-04-01","expected_delivery_date":"2026-04-05","actual_delivery_date":"2026-04-17"},
    {"order_date":"2026-05-01","expected_delivery_date":"2026-05-05","actual_delivery_date":"2026-05-19"},
]


# ══════════════════════════════════════════════════════
# 🧪 6. MIXED EDGE CASE (unstable switching behavior)
# ══════════════════════════════════════════════════════
history_mixed = [
    {"order_date":"2026-01-01","expected_delivery_date":"2026-01-06","actual_delivery_date":"2026-01-07"},
    {"order_date":"2026-02-01","expected_delivery_date":"2026-02-06","actual_delivery_date":"2026-02-10"},
    {"order_date":"2026-03-01","expected_delivery_date":"2026-03-06","actual_delivery_date":"2026-03-05"},
    {"order_date":"2026-04-01","expected_delivery_date":"2026-04-06","actual_delivery_date":"2026-04-12"},
    {"order_date":"2026-05-01","expected_delivery_date":"2026-05-06","actual_delivery_date":"2026-05-04"},
]


# ══════════════════════════════════════════════════════
# 🚨 7. INVALID (TOO SHORT)
# ══════════════════════════════════════════════════════
history_invalid = history_perfect[:2]


# ══════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════
print_result("🟢 PERFECT SUPPLIER (RULE EXPECTED)", history_perfect)
print_result("🟡 NORMAL SUPPLIER (STAT EXPECTED)", history_normal)
print_result("🔴 LATE SUPPLIER (ML EXPECTED)", history_late)
print_result("🔥 CHAOTIC SUPPLIER (ML EXPECTED)", history_chaotic)
print_result("⚠️ EXTREME LATE SUPPLIER", history_extreme_late)
print_result("🧪 MIXED SUPPLIER (EDGE CASE)", history_mixed)

print("\n🚨 INVALID CASE")
print(predict_delay(history_invalid))