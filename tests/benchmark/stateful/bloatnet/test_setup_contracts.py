"""Deploy the CREATE2 contracts assumed to exist by bloatnet benchmarks."""

from execution_testing import (
    DETERMINISTIC_FACTORY_ADDRESS,
    Account,
    Alloc,
    AuthorizationTuple,
    BenchmarkTestFiller,
    Fork,
    Hash,
    Op,
    Transaction,
)

from tests.benchmark.stateful.helpers import (
    AccountMode,
    account_mode_initcode,
    account_mode_runtime_size,
    diff_delegate_authority,
    diff_delegate_target,
    pack_transactions_into_blocks,
)

# EIP-7702 delegation designation prefix (0xef0100 || address).
DELEGATION_PREFIX = bytes([0xEF, 0x01, 0x00])

# Authorities authorized per set-code transaction. Batching amortizes the
# per-transaction intrinsic cost over many authorizations.
DELEGATES_PER_TX = 100

# Distinct contracts deployed per account mode. Sized for the most
# demanding consumer, test_account_access, whose cheapest iteration is
# ~2,691 gas: a 300M block reaches ~111,482 distinct salts, rounded up
# with margin.
RECEIVER_CONTRACT_COUNT = 120_000

CONTRACT_MODES = [
    AccountMode.EXISTING_CONTRACT_MINIMAL,
    AccountMode.EXISTING_CONTRACT_SAME,
    AccountMode.EXISTING_CONTRACT_DIFF,
    AccountMode.EXISTING_CONTRACT_JUMPDEST,
]


def deployment_gas_limit(
    fork: Fork, initcode: bytes, runtime_size: int
) -> int:
    """
    Return the gas limit for one CREATE2 deployment, derived from the
    intrinsic, CREATE2, and code-deposit costs with a small margin.
    """
    intrinsic = fork.transaction_intrinsic_cost_calculator()(
        calldata=Hash(0) + initcode
    )
    create_cost = Op.CREATE2(
        value=0,
        offset=0,
        size=len(initcode),
        salt=0,
        init_code_size=len(initcode),
    ).gas_cost(fork)
    deposit_cost = Op.RETURN(
        0,
        runtime_size,
        code_deposit_size=runtime_size,
    ).gas_cost(fork)
    base = intrinsic + create_cost + deposit_cost
    return base + base // 16


def test_deploy_existing_contracts(
    benchmark_test: BenchmarkTestFiller,
    pre: Alloc,
    fork: Fork,
    gas_benchmark_value: int,
) -> None:
    """
    Deploy the contracts behind the `AccountMode.EXISTING_CONTRACT_*`
    receivers via the deterministic CREATE2 factory.

    Runs once as a global setup before the bloatnet benchmarks; the
    transactions intentionally carry no test phase so that the payload
    pipeline replays them ahead of every scenario. Each deployment's
    gas limit is computed from its cost, then transactions are packed
    into blocks up to the block gas budget.

    After the contracts, deterministic EOAs are delegated via EIP-7702 to the
    EXISTING_CONTRACT_DIFF receivers (delegate ``i`` -> ``i``-th DIFF
    contract). The delegations are part of this setup so that, by the time the
    benchmarks run, the delegated authorities already exist in the chain's
    prestate; the client under test does not create them during the measured
    run and therefore cannot warm/cache them. Benchmarks target these accounts
    by deriving them; see ``helpers.diff_delegate_authority`` /
    ``diff_delegate_target``.
    """
    txs = []
    post = {}
    for account_mode in CONTRACT_MODES:
        initcode = account_mode_initcode(fork, account_mode)
        gas_limit = deployment_gas_limit(
            fork, initcode, account_mode_runtime_size(fork, account_mode)
        )
        sender = pre.fund_eoa()
        for salt in range(RECEIVER_CONTRACT_COUNT):
            txs.append(
                Transaction(
                    to=DETERMINISTIC_FACTORY_ADDRESS,
                    data=Hash(salt) + initcode,
                    gas_limit=gas_limit,
                    sender=sender,
                )
            )

    # ---- EIP-7702 delegates for the EXISTING_CONTRACT_DIFF receivers ----
    # Delegate EOA i (helpers.diff_delegate_authority(i)) is authorized to
    # delegate to the i-th EXISTING_CONTRACT_DIFF contract
    # (helpers.diff_delegate_target(fork, i)). Authorities need no balance; the
    # funded delegation_sender pays. These run in the same setup so the
    # authorities are in the prestate before the measured benchmarks.
    delegation_sender = pre.fund_eoa()
    intrinsic = fork.transaction_intrinsic_cost_calculator()
    for start in range(0, RECEIVER_CONTRACT_COUNT, DELEGATES_PER_TX):
        count = min(DELEGATES_PER_TX, RECEIVER_CONTRACT_COUNT - start)
        auth_list = [
            AuthorizationTuple(
                address=diff_delegate_target(fork, i),
                nonce=0,
                signer=diff_delegate_authority(i),
            )
            for i in range(start, start + count)
        ]
        tx_gas = intrinsic(authorization_list_or_count=count) + 50_000
        txs.append(
            Transaction(
                to=delegation_sender,
                gas_limit=tx_gas,
                sender=delegation_sender,
                authorization_list=auth_list,
            )
        )

    # Each authorized authority ends with its delegation designation as code
    # (0xef0100 || target) and nonce incremented to 1.
    for i in (0, RECEIVER_CONTRACT_COUNT - 1):
        post[diff_delegate_authority(i)] = Account(
            nonce=1,
            code=DELEGATION_PREFIX + bytes(diff_delegate_target(fork, i)),
        )

    blocks = pack_transactions_into_blocks(txs, gas_benchmark_value)

    benchmark_test(
        post=post,
        blocks=blocks,
        skip_gas_used_validation=True,
        expected_receipt_status=1,
    )
