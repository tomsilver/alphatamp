;; Blocks-world domain -- STRIPS subset of policy-guided-lazy-tamp/experiments/blocks_world/domain.pddl
;; Shared by all problem types (sorting, stacking, ...); the goal distinguishes them.
;;
;; The geometric preconditions of the original (ik, colfree-block, table-support,
;; the per-other-block `forall` collision check) are DELIBERATELY dropped here.
;; They are not symbolic: in the LAZY/PIGINet pipeline they are certified by
;; streams during *refinement*. Keeping them out of the symbolic model is exactly
;; what turns "a taller blocker is in the way" into a refinement (geometric)
;; failure rather than a planning failure -- which is the feasibility signal a
;; plan-feasibility predictor (PIGINet) is meant to learn.
;;
;; Single arm (Panda) is modelled with a 0-ary (handempty)/(holding ?b) pair so the
;; domain stays pure STRIPS and is accepted by both SymK and pyperplan.
(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block table)

  (:predicates
    (on-table ?b - block ?t - table)   ; fluent
    (on-block ?b - block ?lb - block)  ; fluent (stacking; sorting goals are flat)
    (clear ?b - block)                 ; fluent
    (holding ?b - block)               ; fluent
    (handempty))                       ; fluent

  ;; pick a clear block off a table
  (:action pick
    :parameters (?b - block ?t - table)
    :precondition (and (on-table ?b ?t) (clear ?b) (handempty))
    :effect (and (holding ?b) (not (on-table ?b ?t)) (not (handempty)) (not (clear ?b))))

  ;; place the held block onto a table
  (:action place
    :parameters (?b - block ?t - table)
    :precondition (holding ?b)
    :effect (and (on-table ?b ?t) (handempty) (clear ?b) (not (holding ?b))))

  ;; lift a clear block off the block beneath it
  (:action unstack
    :parameters (?b - block ?lb - block)
    :precondition (and (on-block ?b ?lb) (clear ?b) (handempty))
    :effect (and (holding ?b) (clear ?lb) (not (on-block ?b ?lb))
                 (not (handempty)) (not (clear ?b))))

  ;; stack the held block onto a clear block
  (:action stack
    :parameters (?b - block ?lb - block)
    :precondition (and (holding ?b) (clear ?lb))
    :effect (and (on-block ?b ?lb) (handempty) (clear ?b)
                 (not (holding ?b)) (not (clear ?lb)))))
