(define (domain drawer_declutter)
  ;; DD2D -- Drawer Decluttering in 2D (docs/dd2d_spec.md Section 6).
  ;; A geometry-BLIND STRIPS domain: the collision/packing structure (which items block
  ;; the target's grasp, whether a staged subset fits the buffer) is deliberately dropped
  ;; from the symbolic model and certified only in refinement (spec Section 6.1 / 3.3).
  ;; Consequence: the shortest optimistic plan is literally ``retrieve(target)`` -- "just
  ;; grab it" -- which fails when the target is blocked, after which the planner grows
  ;; longer staging plans. This is why the DD2D candidate enumerator (which knows the
  ;; blocking geometry) is needed over generic top-k -- see blocks_tamp/dd2d/planning.py.
  (:requirements :strips :typing)
  (:types item)
  (:predicates
    (in-drawer ?o - item)
    (on-buffer ?o - item)
    (holding ?o - item)
    (handempty)
    (target ?o - item)
    (extracted ?o - item))

  (:action pick
    :parameters (?o - item)
    :precondition (and (in-drawer ?o) (handempty))
    :effect (and (holding ?o) (not (in-drawer ?o)) (not (handempty))))

  (:action place-buffer
    :parameters (?o - item)
    :precondition (holding ?o)
    :effect (and (on-buffer ?o) (handempty) (not (holding ?o))))

  (:action retrieve
    :parameters (?o - item)
    :precondition (and (handempty) (target ?o) (in-drawer ?o))
    :effect (and (extracted ?o) (not (in-drawer ?o)))))
