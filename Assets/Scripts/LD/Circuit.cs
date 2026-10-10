using Dreamteck.Splines;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Circuit : MonoBehaviour
{
    [SerializeField] private List<Road> roads = new List<Road>();
    [SerializeField] private Transform spawnPoint;

    public delegate void OnCircuitFinished(CarMovement agent, Circuit circuit);
    public OnCircuitFinished onCircuitFinished;

    public delegate void OnCheckpointCrossed(CarMovement agent, Circuit circuit);
    public OnCheckpointCrossed onCheckpointCrossed;

    public float GetDistanceFromRoad(Transform car)
    {
        return roads.Min(e => e.GetDistanceToSpline(car));
    }

    public Vector3 GetClosestPointFromRoad(Transform car)
    {
        Vector3 closestPoint = Vector3.zero;
        float minDistance = float.MaxValue;

        foreach (var road in roads)
        {
            SplineComputer spline = road.GetSpline();
            SplineSample projection = spline.Project(car.position);
            float distanceFromSpline = Vector3.Distance(car.position, projection.position);
            if(distanceFromSpline <= minDistance)
            {
                minDistance = distanceFromSpline;
                closestPoint = projection.position;
            }
        }

        return closestPoint;
    }

    public Road GetClosestRoad(Transform car)
    {
        Road closestRoad = null;
        float minDistance = float.MaxValue;

        foreach (var road in roads)
        {
            SplineComputer spline = road.GetSpline();
            SplineSample projection = spline.Project(car.position);
            float distanceFromSpline = Vector3.Distance(car.position, projection.position);
            if (distanceFromSpline <= minDistance)
            {
                minDistance = distanceFromSpline;
                closestRoad = road;
            }
        }

        return closestRoad;
    }

    public Vector3 GetSplinePointInFront(float distance, Transform car)
    {
        Road road = GetClosestRoad(car);
        if (road)
        {
            int roadIndex = roads.IndexOf(road);
            SplineComputer spline = road.GetSpline();

            //Travel walks the real distance along the spline (the percent is not proportional to the distance)
            double percentInFront = spline.Travel(spline.Project(car.position).percent, distance, out float moved);
            float remainingDistance = distance - moved;

            //Continue on the next roads while the target is past the end of the current one
            while (remainingDistance > 0.001f && roadIndex < roads.Count - 1)
            {
                roadIndex++;
                spline = roads[roadIndex].GetSpline();
                percentInFront = spline.Travel(0.0, remainingDistance, out moved);
                remainingDistance -= moved;
            }

            return spline.Evaluate(percentInFront).position;
        }

        return Vector3.zero;
    }

    //Distance from the start of the circuit, measured along the road splines
    public float GetDistanceAlongCircuit(Transform car)
    {
        float distanceAlongCircuit = 0f;
        float roadsLengthBefore = 0f;
        float minDistance = float.MaxValue;

        foreach (var road in roads)
        {
            SplineComputer spline = road.GetSpline();
            SplineSample projection = spline.Project(car.position);
            float distanceFromSpline = Vector3.Distance(car.position, projection.position);
            if (distanceFromSpline <= minDistance)
            {
                minDistance = distanceFromSpline;
                distanceAlongCircuit = roadsLengthBefore + spline.CalculateLength(0.0, projection.percent);
            }

            roadsLengthBefore += spline.CalculateLength();
        }

        return distanceAlongCircuit;
    }

    public List<Road> GetRoads()
    {
        return roads; 
    }

    public void SpawnAgent(CarMovement agent)
    {
        agent.GetRigidbody().MovePosition(spawnPoint.position);
        agent.transform.position = spawnPoint.position;
        agent.car.transform.rotation = spawnPoint.rotation;
    }

    public void ResetCircuit()
    {
        foreach (var road in roads)
        {
            RandomizeSplinePoint randomize = road.GetComponent<RandomizeSplinePoint>();
            if (randomize != null)
            {
                randomize.Randomize();
            }
        }
    }

    public Transform GetCarSpawnPoint()
    {
        return spawnPoint;
    }
}
