"""
Credit the deterministic benchmark sender pool via consensus-layer withdrawals.

The stateful repricing benchmarks draw their senders from
`yield_distinct_sender()`, whose i-th account is the EOA of private key
`SENDER_BASE_KEY + i`. Those accounts are deliberately kept out of the
pre-allocation so they stay uncached, which means nothing funds them: a
benchmark that uses them on a fresh snapshot fails with

    insufficient funds for gas * price + value:
    address 0x4e5e4CBB5d1c13242118aA32f02c7723D9c9377a have 0 want 11040001

(that address being sender index 0). `test_setup_contracts.py` deploys the
receiver contracts and the EIP-7702 delegations but funds nothing, so this test
is the missing companion to it.

Withdrawals are used rather than transactions for two reasons: they credit
balance without executing anything, so the senders stay uncached until the
benchmark under measurement first touches them; and they cost no gas, so the
funding blocks do not consume any of the block's benchmark budget.

Run it on a snapshot that already carries a large block gas limit (jochemnet has
1 TGas in its head), so no gas ramp is needed -- only these withdrawal blocks.

Not marked `repricing`: it is prestate setup, not a benchmark, and must stay out
of `-m repricing` selection.
"""

import itertools

import pytest
from execution_testing import (
    Alloc,
    BenchmarkTestFiller,
    Block,
    Withdrawal,
)

from tests.benchmark.helper.account_sender_receiver import (
    yield_distinct_sender,
)

# Senders to credit. The per-variant floor is set by the biggest gas target over
# the cheapest per-iteration cost -- 300,000,000 / ~21,000 gas per ether transfer
# ~= 14,286 -- but the pool is sized well past that so a single funded prestate
# serves higher gas targets and cheaper per-iteration variants without refunding.
# Matches the 150k pool of the jochemnet repricing prestate.
FUNDED_SENDER_COUNT = 150_000

# Max uint64 gwei (~18.4M ETH) each -- far more than any benchmark can spend, and
# the same amount NethermindEth/gas-benchmarks' funding block used.
WITHDRAWAL_AMOUNT_GWEI = 2**64 - 1

# One block per WITHDRAWALS_PER_BLOCK. A withdrawal is ~150 B of payload, so
# 15,000 of them is a ~2.2 MB block -- comfortably inside the EIP-7934 block RLP
# size limit. 150,000 senders therefore take 10 blocks. Do not raise this to fit
# the pool in one block: the RLP limit is the binding constraint, not the gas
# limit, since withdrawals cost no gas.
WITHDRAWALS_PER_BLOCK = 15_000


@pytest.mark.valid_from("Amsterdam")
def test_fund_sender_pool(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
) -> None:
    """Credit `FUNDED_SENDER_COUNT` deterministic senders by withdrawal."""
    recipients = list(
        itertools.islice(yield_distinct_sender(), FUNDED_SENDER_COUNT)
    )
    withdrawals = [
        Withdrawal(
            index=index,
            validator_index=index,
            address=recipient,
            amount=WITHDRAWAL_AMOUNT_GWEI,
        )
        for index, recipient in enumerate(recipients)
    ]

    benchmark_test(
        pre=pre,
        post={},
        blocks=[
            Block(
                txs=[],
                withdrawals=withdrawals[index : index + WITHDRAWALS_PER_BLOCK],
            )
            for index in range(0, len(withdrawals), WITHDRAWALS_PER_BLOCK)
        ],
        expected_benchmark_gas_used=0,
    )
