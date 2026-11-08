"""Parent-owned reply resources with bounded, explicitly retired lifetimes.

Workers receive only the existing spawn-safe lease. Retiring a lease requires
its workers to have stopped. In-flight inference keeps shared views pinned until
the service finishes the envelope; retired lane IDs are never reused.
"""
from collections import defaultdict
import threading

from brass_ai.learning.contracts import require


class ReplyChannels:
    def __init__(self, mp, slot_bytes):
        self.mp,self.slot_bytes=mp,slot_bytes
        self.connections={};self.children=[];self.slots={}
        self._children={};self._users={};self._retired=set()
        self.lock=threading.RLock();self.next_lane=0;self.created=0;self.released=0

    def allocate(self):
        from .shared_transport import SharedSlot
        with self.lock:
            parent,child=self.mp.Pipe(duplex=True);slot=None
            try:
                if self.slot_bytes:slot=SharedSlot(size=self.slot_bytes)
            except BaseException:
                parent.close();child.close();raise
            lane=self.next_lane;self.next_lane+=1;self.created+=1
            self.connections[lane]=parent;self.children.append(child);self._children[lane]=child
            self._users[lane]=0
            if slot is not None:self.slots[lane]=slot
            return child,lane,slot.description if slot is not None else None

    def pin(self,lane):
        with self.lock:
            require(type(lane) is int and 0<=lane<self.next_lane,'Unknown private inference lane')
            if lane in self._retired or lane not in self.connections:return False
            self._users[lane]+=1;return True

    def retired(self,lane):
        with self.lock:return lane in self._retired or lane not in self.connections

    def unpin(self,lane):
        with self.lock:
            require(self._users.get(lane,0)>0,'Inference lane ownership underflow')
            self._users[lane]-=1
            if lane in self._retired and not self._users[lane]:self._dispose(lane)

    def retire(self,lanes):
        with self.lock:
            for lane in lanes:
                if lane not in self.connections:continue
                self._retired.add(lane)
                # Close the parent's worker-end copy so a dead reader cannot
                # leave the inference thread blocked while sending its reply.
                self._children[lane].close()
                if not self._users[lane]:self._dispose(lane)

    def _dispose(self,lane):
        self.connections.pop(lane).close()
        child=self._children.pop(lane);child.close();self.children.remove(child)
        slot=self.slots.pop(lane,None)
        if slot is not None:slot.close()
        self._users.pop(lane);self._retired.discard(lane);self.released+=1

    def close_children(self):
        with self.lock:
            for child in self.children:child.close()

    def close(self):
        """Only after every service using this registry has stopped/joined."""
        with self.lock:
            for lane in list(self.connections):self._dispose(lane)

    def report(self):
        with self.lock:
            return dict(created=self.created,released=self.released,active=len(self.connections),
                retiring=len(self._retired),shared_slots=len(self.slots))


def allocate_lease(owner,count,lease_type,allocate):
    require(type(count) is int and count>0,'Missing inference reply lanes')
    require(not owner.closed,'Inference service is closed')
    resources=defaultdict(list);items=[]
    try:
        # Allocate transactionally before queueing any transferable handles.
        # Partial Pipe/SharedMemory/Queue creation must not strand descriptors.
        for _ in range(count):
            registry,lane,item=allocate();resources[registry].append(lane);items.append(item)
        leases=owner.mp.Queue()
    except BaseException:
        for registry,lanes in resources.items():registry.retire(lanes)
        raise
    lease=lease_type(leases,owner.requests if lease_type.connection_index==0 else tuple(s.requests for s in owner.services))
    owner.leases.append(leases);owner._lease_channels[leases]=(lease,resources)
    try:
        for item in items:leases.put(item)
    except BaseException:
        release_lease(owner,lease);raise
    return lease


def release_lease(owner,lease):
    """Caller has joined all consumers; drain unclaimed FD transfers as well."""
    if lease.closed:return
    require(lease.leases in owner._lease_channels,'Reply lease belongs to another service')
    leases=lease.leases
    # The marker follows this parent's feeder writes. Queue.empty()/get_nowait()
    # can miss those writes and leak multiprocessing's duplicated FD transfers.
    leases.put(None)
    while True:
        item=leases.get(timeout=30)
        if item is None:break
        item[lease.connection_index].close()
    leases.close();leases.join_thread()
    _,resources=owner._lease_channels.pop(leases);owner.leases.remove(leases);lease.closed=True
    for registry,lanes in resources.items():registry.retire(lanes)


def release_all(owner):
    # The lease wrapper is local; worker copies never own service resources.
    for lease,_ in list(owner._lease_channels.values()):release_lease(owner,lease)
