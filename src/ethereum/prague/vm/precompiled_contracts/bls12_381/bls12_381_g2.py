"""
Ethereum Virtual Machine (EVM) BLS12 381 G2 CONTRACTS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Implementation of pre-compiles in G2 (curve over base prime field).
"""
from py_ecc.bls12_381.bls12_381_curve import add, multiply

from ethereum.base_types import U256, Uint

from ....vm import Evm
from ....vm.gas import charge_gas
from ....vm.memory import buffer_read
from ...exceptions import InvalidParameter
from . import (
    K_DISCOUNT,
    MAX_DISCOUNT,
    MULTIPLIER,
    G2_to_bytes,
    bytes_to_G2,
    decode_G2_scalar_pair,
)

LEN_PER_PAIR = 288


def bls12_g2_add(evm: Evm) -> None:
    """
    The bls12_381 G2 point addition precompile.

    Parameters
    ----------
    evm :
        The current EVM frame.

    Raises
    ------
    InvalidParameter
        If the input length is invalid.
    """
    data = evm.message.data
    if len(data) != 512:
        raise InvalidParameter("Invalid Input Length")

    # GAS
    charge_gas(evm, Uint(800))

    # OPERATION
    p1 = bytes_to_G2(buffer_read(data, U256(0), U256(256)))
    p2 = bytes_to_G2(buffer_read(data, U256(256), U256(256)))

    result = add(p1, p2)

    evm.output = G2_to_bytes(result)


def bls12_g2_multiply(evm: Evm) -> None:
    """
    The bls12_381 G2 multiplication precompile.

    Parameters
    ----------
    evm :
        The current EVM frame.

    Raises
    ------
    InvalidParameter
        If the input length is invalid.
    """
    data = evm.message.data
    if len(data) != LEN_PER_PAIR:
        raise InvalidParameter("Invalid Input Length")

    # GAS
    charge_gas(evm, Uint(45000))

    # OPERATION
    p, m = decode_G2_scalar_pair(data)
    result = multiply(p, m)

    evm.output = G2_to_bytes(result)


def bls12_g2_msm(evm: Evm) -> None:
    """
    The bls12_381 G2 multi-scalar multiplication precompile.
    Note: This uses the naive approach to multi-scalar multiplication
    which is not suitably optimized for production clients. Clients are
    required to implement a more efficient algorithm such as the Pippenger
    algorithm.

    Parameters
    ----------
    evm :
        The current EVM frame.

    Raises
    ------
    InvalidParameter
        If the input length is invalid.
    """
    data = evm.message.data
    if len(data) == 0 or len(data) % LEN_PER_PAIR != 0:
        raise InvalidParameter("Invalid Input Length")

    # GAS
    k = len(data) // LEN_PER_PAIR
    if k <= 128:
        discount = K_DISCOUNT[k]
    else:
        discount = MAX_DISCOUNT

    gas_cost = Uint(k * 45000 * discount // MULTIPLIER)
    charge_gas(evm, gas_cost)

    # OPERATION
    for i in range(k):
        start_index = i * LEN_PER_PAIR
        end_index = start_index + LEN_PER_PAIR

        p, m = decode_G2_scalar_pair(data[start_index:end_index])
        product = multiply(p, m)

        if i == 0:
            result = product
        else:
            result = add(result, product)

    evm.output = G2_to_bytes(result)
