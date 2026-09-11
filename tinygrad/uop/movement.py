from tinygrad.uop.ops import PatternMatcher, UPat, Ops, GroupOp, UOp

# movement op on INDEX as a PatternMatcher
def _mop_index(r:UOp, idx:UOp):
  from tinygrad.schedule.indexing import apply_movement_op
  idxs = idx.src[1:]
  if len(idxs) == len(r.shape):
    return r.src[0].index(*apply_movement_op(r.op, r.src[0].shape, r.marg, idxs), arg=idx.arg)
  if r.op is Ops.RESHAPE:
    src_prefix = len(r.src[0].shape) - len(r.shape[len(idxs):])
    if src_prefix >= 0 and r.src[0].shape[src_prefix:] == r.shape[len(idxs):]:
      if src_prefix == 0: return r.src[0]
      ret = r.src[0].index(*apply_movement_op(r.op, r.src[0].shape[:src_prefix], r.shape[:len(idxs)], idxs), arg=idx.arg)
      return ret if ret.shape == idx.shape else None

pm_mops = PatternMatcher([
  # handle movement ops on INDEX
  (UPat(GroupOp.Movement, name="r").f(Ops.INDEX, allow_any_len=True, name="idx"), _mop_index),
  # move movement ops and INDEX after AFTER
  (UPat(GroupOp.Movement|{Ops.INDEX}, name="r").after(name="a", allow_any_len=True),
   lambda r,a: UOp(r.op, src=(a.replace(src=(r.src[0],)+a.src[1:]),)+r.src[1:], arg=r.arg)),
  (UPat(GroupOp.Movement, name="r").end(name="a", allow_any_len=True), lambda r,a: a.replace(src=(r.src[0],)+a.src[1:])),
])

mop_cleanup = PatternMatcher([
  # merge adjacent RESHAPES
  (UPat(Ops.RESHAPE, src=(UPat(Ops.RESHAPE, name="x2"), UPat()), name="x"), lambda x,x2: x.replace(src=(x2.src[0], x.src[1]))),
  # remove noop RESHAPEs
  (UPat(Ops.RESHAPE, src=(UPat(name="x2"), UPat()), name="x"), lambda x,x2: x2 if x2._shape is not None and x2.shape == x.shape else None),
  # merge PERMUTEs
  (UPat(Ops.PERMUTE, src=(UPat(Ops.PERMUTE, name="x2"),), name="x"), lambda x,x2: x2.replace(arg=tuple(x2.arg[i] for i in x.arg))),
  # remove noop PERMUTEs
  (UPat(Ops.PERMUTE, name="x"), lambda x: x.src[0] if list(x.arg) == list(range(len(x.arg))) else None),
  # STACK on INDEX CONST
  (UPat(Ops.STACK, src=UPat(Ops.INDEX, src=(UPat.var("src"), UPat(Ops.CONST))), name="stk"),
   lambda src,stk: src if stk.shape == src.shape and list(range(len(stk.src))) == [x.src[1].val for x in stk.src] else None),
  # const INDEX into STACK is src
  (UPat(Ops.INDEX, src=(UPat(Ops.STACK, name="a"), UPat.cvar("i")), name="idx", allow_any_len=True),
   lambda a,i,idx: a.src[i.val] if len(idx.src) <= 2 else a.src[i.val].index(*idx.src[2:])),
  # INDEX on INDEX is INDEX
  (UPat(Ops.INDEX, src=(UPat(Ops.INDEX, name="idx1", allow_any_len=True),), allow_any_len=True, name="idx2"),
   lambda idx1,idx2: idx1.src[0].index(*idx1.src[1:], *idx2.src[1:]) if all(x.shape == () for x in idx1.src[1:]+idx2.src[1:]) else None),
  # INDEX on shaped INDEX (TODO: this can be more generic)
  (UPat(Ops.INDEX, src=(UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("idx1_arg"))),), allow_any_len=True, name="idx2"),
   lambda buf,idx1_arg,idx2: buf.index(idx1_arg.index(*idx2.src[1:])) if len(idx1_arg.shape) == len(idx2.src[1:]) else None),
])
