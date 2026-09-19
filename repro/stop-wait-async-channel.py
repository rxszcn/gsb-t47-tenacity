import asyncio

import tenacity
from tenacity import stop_after_attempt, stop_all, stop_any, wait_combine, wait_fixed


def make_astop(threshold):
    async def astop(rs):
        return rs.attempt_number >= threshold

    return astop


async def run(stop_obj, label, cap=10):
    n = {"c": 0}

    async def f():
        n["c"] += 1
        if n["c"] >= cap:
            raise RuntimeError("撞到 %d 次保险上限" % cap)
        raise ValueError("x")

    inst = tenacity.AsyncRetrying(
        stop=stop_obj,
        wait=tenacity.wait_none(),
        sleep=lambda s: asyncio.sleep(0),
        reraise=True,
    )
    try:
        await inst(f)
    except BaseException as e:  # noqa: BLE001
        err = type(e).__name__
    else:
        err = "-"
    print("   %-52s 尝试=%-3d 终态=%s" % (label, n["c"], err))


async def main():
    print("=== 1) 同一个协程停止条件（第 4 次才该停），裸传 vs 进组合器")
    await run(make_astop(4), "stop=<协程函数>")
    await run(stop_any(make_astop(4)), "stop=stop_any(<协程函数>)")
    await run(stop_any(make_astop(4), stop_after_attempt(99)), "stop=stop_any(<协程函数>, 99 次上限)")
    await run(stop_any(make_astop(4), tenacity.stop_never), "stop=stop_any(<协程函数>, stop_never)")
    print()
    print("=== 2) 同步成员单独用是对照组（证明不是 AsyncRetrying 的问题）")
    await run(stop_after_attempt(4), "stop=stop_after_attempt(4)")
    await run(stop_any(stop_after_attempt(4), stop_after_attempt(99)), "stop=stop_any(同步4, 同步99)")
    await run(stop_all(stop_after_attempt(4), stop_after_attempt(2)), "stop=stop_all(同步4, 同步2)")

    print()
    print("=== 3) wait 侧：协程等待函数裸传 ok，进组合器就崩")
    async def await_15(rs):
        return 1.5

    def mk_slept():
        box = []

        def s(sec):
            box.append(sec)
            return asyncio.sleep(0)

        return box, s

    box, s = mk_slept()
    calls = {"c": 0}

    async def f2():
        calls["c"] += 1
        if calls["c"] < 3:
            raise ValueError("x")
        return "ok"

    inst = tenacity.AsyncRetrying(stop=stop_after_attempt(3), wait=await_15, sleep=s)
    print("   wait=<协程函数> 结果=%r 记录的等待值=%r" % (await inst(f2), box))

    for label, w in [
        ("wait_combine(wait_fixed(1), <协程函数>)", wait_combine(wait_fixed(1), await_15)),
        ("wait_fixed(1) + <协程函数>", wait_fixed(1) + await_15),
        ("wait_chain 之外的 stop_before_delay+协程", None),
    ]:
        if w is None:
            continue
        box2, s2 = mk_slept()
        calls["c"] = 0
        try:
            r = await tenacity.AsyncRetrying(
                stop=stop_after_attempt(3), wait=w, sleep=s2
            )(f2)
            print("   %-40s 结果=%r 等待值=%r" % (label, r, box2))
        except BaseException as e:  # noqa: BLE001
            print("   %-40s -> %s: %s" % (label, type(e).__name__, str(e)[:60]))

    print()
    print("=== 4) 异步侧根本没有 stop/wait 模块；组合器的可等待性")
    import tenacity.asyncio as ta

    print("   tenacity.asyncio 中 stop*/wait* 名字:",
          [x for x in dir(ta) if x.startswith(("stop", "wait"))])
    print("   is_coroutine_callable(stop_any(协程).__call__) ->",
          tenacity._utils.is_coroutine_callable(stop_any(make_astop(4)).__call__))
    print("   stop_any(协程)(state) 的返回值 ->", end=" ")
    rs = tenacity.RetryCallState(retry_object=None, fn=None, args=(), kwargs={})
    rs.attempt_number = 1
    v = stop_any(make_astop(4))(rs)
    print("%s  bool(%s)=%s" % (type(v).__name__, v, bool(v)))
    if hasattr(v, 'close'):
        v.close()


asyncio.run(main())
