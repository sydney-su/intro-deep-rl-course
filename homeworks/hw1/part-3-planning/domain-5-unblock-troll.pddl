
(define (domain action-castle)
   (:requirements :strips :typing)
   (:types player location direction monster item fishingpole food)

   (:action go
      :parameters (?dir - direction ?p - player ?l1 - location ?l2 - location)
      :precondition (and (at ?p ?l1) (connected ?l1 ?dir ?l2) (not (blocked ?l1 ?dir ?l2)))
      :effect (and (at ?p ?l2) (not (at ?p ?l1)))
   )

   (:action get
      :parameters (?obj - item ?p - player ?l - location)
      :precondition (and (at ?p ?l) (at ?obj ?l) (not (fishable ?obj)))
      :effect (and (inventory ?p ?obj) (not (at ?obj ?l)))
   )

   (:action drop
      :parameters (?obj - item ?p - player ?l - location)
      :precondition (and (inventory ?p ?obj) (at ?p ?l))
      :effect (and (not (inventory ?p ?obj)) (at ?obj ?l))
   )

   (:action gofish
      :parameters (?pole - fishingpole ?p - player ?l - location ?fish - item)
      :precondition (and (haslake ?l) (at ?fish ?l) (fishable ?fish) (inventory ?p ?pole) (at ?p ?l))
      :effect (not (fishable ?fish))
   )

   (:action feed
      :parameters (?m - monster ?f - food ?p - player ?l - location)
      :precondition (and (at ?p ?l) (at ?m ?l) (inventory ?p ?f) (hungry ?m))
      :effect (and (not (hungry ?m)) (not (inventory ?p ?f)))
   )
   
   (:action feed-troll
      :parameters (?m - monster ?f - food ?p - player ?l1 - location ?dir - direction ?l2 - location)
      :precondition (and (at ?p ?l1) (at ?m ?l1) (inventory ?p ?f) (hungry ?m) (blocked ?l1 ?dir ?l2))
      :effect (and (not (hungry ?m)) (not (inventory ?p ?f)) (not (blocked ?l1 ?dir ?l2)))
   )
)
