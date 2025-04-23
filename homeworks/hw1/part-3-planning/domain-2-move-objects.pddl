
(define (domain action-castle)
   (:requirements :strips :typing)
   (:types player location direction monster item)

   (:action go
      :parameters (?dir - direction ?p - player ?l1 - location ?l2 - location)
      :precondition (and (at ?p ?l1) (connected ?l1 ?dir ?l2) (not (blocked ?l1 ?dir ?l2)))
      :effect (and (at ?p ?l2) (not (at ?p ?l1)))
   )

   (:action get
      :parameters (?obj - item ?p - player ?l - location)
      :precondition (and (at ?p ?l) (at ?obj ?l))
      :effect (and (inventory ?p ?obj) (not (at ?obj ?l)))
   )

   (:action drop
      :parameters (?obj - item ?p - player ?l - location)
      :precondition (and (inventory ?p ?obj) (at ?p ?l))
      :effect (and (not (inventory ?p ?obj)) (at ?obj ?l))
   )
)
