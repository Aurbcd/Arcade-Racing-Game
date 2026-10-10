using UnityEngine;
using System.Collections.Generic;
using Evaluation;

public class ObstacleTracker : MonoBehaviour
{
    [SerializeField] private int obstacleLayer;
    [SerializeField] private float maxObstaclePenaltyReturn = -10f;
    [SerializeField][Tooltip("Given once per obstacle hit in the episode, on top of the penalty while in contact")] private float obstacleHitPenalty = -0.75f;

    [Header("References")]
    [SerializeField] private CarMovement agent;

    private int contactObstacleCount = 0;
    private List<GameObject> hitObstacles = new List<GameObject>();
    public bool IsInContactWithObstacles => contactObstacleCount > 0;

    public void ResetObstacleTracker()
    {
        contactObstacleCount = 0;
        hitObstacles.Clear();
    }

    private void FixedUpdate()
    {
        EvaluationTests.SetAgentIsColliding(IsInContactWithObstacles);

        if(IsInContactWithObstacles)
        {
            agent.AddReward(RewardTerm.ObstacleContact, maxObstaclePenaltyReturn / agent.MaxStep);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == obstacleLayer)
        {
            contactObstacleCount++;
            if(!hitObstacles.Contains(collision.gameObject))
            {
                hitObstacles.Add(collision.gameObject);
                EvaluationTests.AddHitObjstacle();
                agent.AddReward(RewardTerm.ObstacleHits, obstacleHitPenalty);
            }
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.layer == obstacleLayer)
        {
            contactObstacleCount--;
            if(contactObstacleCount < 0)
            {
                contactObstacleCount = 0;
            }
        }
    }
}
