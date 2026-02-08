import numpy as np

INTENTS = [
    "needs_reply",
    "meeting_request",
    "invoice_payment",
    "action_required",
    "info_only",
    "newsletter",
]

# Rows = gold, columns = predicted
CM = np.array([
    [0, 2, 0, 1, 2, 0],   # needs_reply
    [0, 5, 0, 0, 7, 0],   # meeting_request
    [6, 0, 6, 0, 14, 0],  # invoice_payment
    [5, 7, 1, 17, 8, 1],  # action_required
    [8, 4, 1, 6, 35, 3],  # info_only
    [2, 5, 0, 1, 42, 11]  # newsletter
])
