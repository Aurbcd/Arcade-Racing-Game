using System;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Evaluation;
using Unity.VisualScripting;

//Reward terms, each summed over the episode and sent to TensorBoard as Reward/<term> (same names as the Simulation project)
public enum RewardTerm { Progress, Time, Checkpoints, Finish, ObstacleContact, ObstacleHits, OffTrack, OffTrackEnd }

public class CarMovement : Agent
{
    private IAM_Car inputActions;

    [Header("References")]
    [SerializeField] private Rigidbody rb;
    [SerializeField] public Transform car;
    [SerializeField] private Transform mesh;
    [SerializeField] private CarCamera carCamera;
    [SerializeField] private CheckpointManager checkpointManager;
    [SerializeField] private ObstacleTracker obstacleTracker;
    [SerializeField] private EnvTraining envTraining;

    [Header("GroundDetection")]
    [SerializeField] private float groundDetectionDistance;
    [SerializeField] private int layerRoad;
    [SerializeField] private LayerMask groundLayerMask;
    private bool isGrounded;
    private bool isOnRoad;
    private Vector3 groundNormal;

    [Header("Gravity")]
    [SerializeField] private float gravityMultGrounded = 1.0f;
    [SerializeField] private float gravityMultInAir = 1.0f;

    [Header("Speed")]
    [SerializeField] private float targetSpeedAcceleration;
    [SerializeField] private float targetSpeedBoost;
    [SerializeField] private float maxSpeed;
    [SerializeField] private float offRoadTargetSpeedMultiplier = 0.5f;

    [Header("Acceleration")]
    [SerializeField] private float maxAcceleration;
    [SerializeField][Tooltip("t=0 => speed=0\nt=1 => speed=targetSpeedAcceleration")] private AnimationCurve accelerationBySpeed;
    private bool isAccelerating;

    [Header("Boost")]
    [SerializeField] private bool isBoostEnabled;
    [SerializeField] private float maxAccelerationBoost;
    [SerializeField][Tooltip("t=0 => speed=0\nt=1 => speed=targetSpeedAcceleration")] private AnimationCurve accelerationBoostBySpeed;
    [SerializeField] private float boostDuration;
    [SerializeField] private float boostDelayBeforeRefill;
    [SerializeField] private float boostRefillDuration;
    private bool wantToBoost;
    private float boostGauge = 1.0f; //[0;1]
    public float BoostGauge
    {
        get { return boostGauge; }
        set
        {
            boostGauge = value;
            onBoostGaugeChanged?.Invoke(boostGauge);
        }
    }
    public static Action<float> onBoostGaugeChanged;
    private Coroutine cor_BoostRefill;

    [Header("Deceleration")]
    [SerializeField] private float maxDeceleration;
    [SerializeField] private float maxDecelerationOffRoad;
    [SerializeField][Tooltip("t=0 => speed=maxSpeed\nt=1 => speed=0")] private AnimationCurve decelerationBySpeed;

    [Header("Brake")]
    [SerializeField] private float maxBrakeDeceleration;
    [SerializeField][Tooltip("t=0 => speed=maxSpeed\nt=1 => speed=0")] private AnimationCurve brakeDecelerationBySpeed;
    private bool isBraking;

    [Header("Allign to Ground")]
    [SerializeField] private float allignRotationSpeed = 1.0f;
    [SerializeField] private float allignRotationSpeedInAir = 1.0f;
    private Vector3 targetCarForward;

    [Header("Turn")]
    [SerializeField] private float speedTurn = 90f;
    [SerializeField] private float turnDampingInAir = 1.0f;
    private bool isTurning = false;
    private float turnInput = 0.0f;
    private float turnMomentum;


    [Header("Drift")]
    [SerializeField] private bool isDriftEnabled;
    [SerializeField] private Vector2 driftInputRemap = new Vector2(0f, 2.0f);
    [SerializeField] private float driftMeshOffsetAngle = 10.0f;
    [SerializeField] private float driftMeshOffsetRotationSpeed = 10.0f;

    [SerializeField] private float driftBoost = 1.5f;
    [SerializeField] private float driftBoostTimeMin = 2.0f;
    [SerializeField] private float driftBoostTimeMax = 5.0f;
    [SerializeField] private float driftBoostDuration = 2.0f;

    private bool isDrifting = false;
    private bool isUsingDriftBoost = false;
    private float driftTime = 0.0f;
    private int driftDirection;
    private Coroutine cor_DriftBoost;

    [Header("Traction")]
    [SerializeField] private float tractionTimeDefault = 0.1f;
    [SerializeField] private float tractionTimeDrift = 0.3f;
    [SerializeField] private float tractionTimeVaritationSpeed = 10f;
    [SerializeField] private float projectVelocityOnGroundSmoothTime = 0.1f;
    private float smoothTimeTraction;
    private Vector3 velProjectOnGround = Vector3.zero;
    private Vector3 smoothVelTraction = Vector3.zero;

    [Header("Rewards")]
    [SerializeField] private float maxTimePenaltyReturn = -1;
    [SerializeField][Tooltip("Reward for each meter of progress along the road splines")] private float progressRewardPerMeter = 0.001f;
    [SerializeField][Tooltip("Max return of the off-track penalty, applied beyond roadHalfWidth from the road centre line")] private float splineDistancePenaltyMultiplier = 1.0f;
    [SerializeField][Tooltip("Ends the episode when the car is further than this from the road centre line")] private float maxDistanceFromSpline = 40.0f;
    [SerializeField] private float outOfTrackPenalty = -2.0f;
    private float timePenaltyReward = 0f;
    private float lastDistanceAlongCircuit = 0f;

    [Header("Agent")]
    [SerializeField] private float accelerationInputDeadZone = 0.25f;
    [SerializeField][Tooltip("Normalizes the offset from the road centre line in the observations, and starts the off-track penalty")] private float roadHalfWidth = 9.0f;
    private int episodeStepCount = 0;
    private float episodeBrakeInputSum = 0f;
    private readonly float[] episodeRewards = new float[Enum.GetValues(typeof(RewardTerm)).Length];

    private const float NORMALIZATION_VALUE_OBSERVATION_VECTORS = 500.0F;
    //Same distances as the Simulation project
    private static readonly float[] LOOK_AHEAD_DISTANCES = { 25f, 50f, 100f, 150f };


    #region Agent
    public override void Initialize()
    {
        inputActions = new IAM_Car();
        inputActions.Enable();

        smoothTimeTraction = tractionTimeDefault;
        timePenaltyReward = maxTimePenaltyReturn / MaxStep;
    }

    public override void OnEpisodeBegin()
    {
        RecordEpisodeStats();

        rb.linearVelocity = Vector3.zero;
        BoostGauge = 1.0f;
        if(cor_BoostRefill != null)
        {
            StopCoroutine(cor_BoostRefill);
            cor_BoostRefill = null;
        }
        if (cor_DriftBoost != null)
        {
            StopCoroutine(cor_DriftBoost);
            cor_DriftBoost = null;
            isUsingDriftBoost = false;
        }

        EvaluationTests.LaunchTrial();

        if (envTraining)
            envTraining.ResetEnvTraining();

        if (checkpointManager)
            checkpointManager.ResetCheckpointManager();

        if(obstacleTracker)
            obstacleTracker.ResetObstacleTracker();

        carCamera.Warp();

        if (envTraining && envTraining.Circuit)
            lastDistanceAlongCircuit = envTraining.Circuit.GetDistanceAlongCircuit(transform);

        EvaluationTests.SetAgentMaxSpeed(maxSpeed);
        EvaluationTests.SetMaxCheckpoint(checkpointManager.CheckpointCount);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(car.InverseTransformDirection(rb.linearVelocity / maxSpeed));

        if (checkpointManager && checkpointManager.GetNextCheckpoint())
        {
            Vector3 toNextCheckpoint = checkpointManager.GetNextCheckpoint().transform.position - transform.position;
            sensor.AddObservation(car.InverseTransformDirection(toNextCheckpoint / NORMALIZATION_VALUE_OBSERVATION_VECTORS));
        }
        else sensor.AddObservation(Vector3.zero);

        sensor.AddObservation(GetGroundID());
        sensor.AddObservation(isBoostEnabled ? BoostGauge : 0f); // 0 without boost, like in Simulation where it is always 0

        if (envTraining.Circuit)
        {
            Vector3 toSpline0m = car.InverseTransformDirection((envTraining.Circuit.GetClosestPointFromRoad(transform) - transform.position) / roadHalfWidth);
            sensor.AddObservation(toSpline0m);

            //Each point is normalized by its own distance, so the far points are as readable as the close ones
            foreach (float distance in LOOK_AHEAD_DISTANCES)
            {
                Vector3 toSplinePoint = car.InverseTransformDirection((envTraining.Circuit.GetSplinePointInFront(distance, transform) - transform.position) / distance);
                sensor.AddObservation(toSplinePoint);
            }
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
            foreach (float distance in LOOK_AHEAD_DISTANCES)
            {
                sensor.AddObservation(Vector3.zero);
            }
        }

        sensor.AddObservation(isGrounded ? groundNormal : Vector3.up); // Changes only in Simulation, in Arcade it will always be Vector3.up
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        AddStepRewards();

        turnInput = actions.ContinuousActions[0];
        isTurning = turnInput != 0.0f;

        float accelerationInput = actions.ContinuousActions[1];
        isBraking = accelerationInput <= -accelerationInputDeadZone;
        isAccelerating = accelerationInput >= accelerationInputDeadZone;

        episodeStepCount++;
        episodeBrakeInputSum += isBraking ? 1.0f : 0.0f;

        if(isDriftEnabled)
        {
            bool isDriftInputPressed = actions.DiscreteActions[0] == 1;
            if(isDrifting ^ isDriftInputPressed)
            {
                if(isDriftInputPressed)
                {
                    OnDriftStart();

                }
                else
                {
                    OnDriftEnd();
                }
            }

            if (isDrifting && driftDirection == 0.0f && isTurning)
            {
                driftDirection = turnInput > 0 ? 1 : -1;
            }
        }

        if (isBoostEnabled)
        {
            bool currentBoost = actions.DiscreteActions[1] == 1;
            if (currentBoost)
            {
                wantToBoost = true;
            }
            else
            {
                wantToBoost = false;
                RefillBoost();
            }
        }

        EvaluationTests.SetAgentTrainingStepValue(StepCount);
        EvaluationTests.SetAgentSpeed(rb.linearVelocity.magnitude);
        EvaluationTests.SetAgentOffRoad(!isOnRoad);
    }

    private void OnDriftStart()
    {
        isDrifting = true;
    }

    private void OnDriftEnd()
    {
        if (driftTime >= driftBoostTimeMin)
        {
            if (cor_DriftBoost != null)
            {
                StopCoroutine(cor_DriftBoost);
            }
            cor_DriftBoost = StartCoroutine(LaunchDriftBoost());
        }

        isDrifting = false;
        driftDirection = 0;
        driftTime = 0f;
    }

    private IEnumerator LaunchDriftBoost()
    {
        driftTime = Mathf.Clamp(driftTime, driftBoostTimeMin, driftBoostTimeMax);
        float ratioBoost = driftTime / driftBoostTimeMax;
        rb.linearVelocity += GetAccelerationDirection() * driftBoost * ratioBoost;
        isUsingDriftBoost = true;
        yield return new WaitForSeconds(driftBoostDuration);
        isUsingDriftBoost = false;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var discreteActionsOut = actionsOut.DiscreteActions;

        continuousActionsOut[0] = inputActions.Car.Turn.ReadValue<float>();

        if(inputActions.Car.Brake.IsPressed())
            continuousActionsOut[1] = -1.0f;
        else if(inputActions.Car.Accelerate.IsPressed())
            continuousActionsOut[1] = 1.0f;
        else
            continuousActionsOut[1] = 0.0f;

        discreteActionsOut[0] = isDriftEnabled && inputActions.Car.Drift.IsPressed() ? 1 : 0;
        discreteActionsOut[1] = isBoostEnabled && inputActions.Car.Boost.IsPressed() ? 1 : 0;
    }

    #endregion

    protected override void OnEnable()
    {
        envTraining.onCircuitUpdated += OnCircuitUpdated;

        base.OnEnable();
    }

    protected override void OnDisable()
    {
        if(envTraining.Circuit != null)
            envTraining.Circuit.onCheckpointCrossed -= OnCheckpointReached;

        envTraining.onCircuitUpdated -= OnCircuitUpdated;

        base.OnDisable();
    }


    private void FixedUpdate()
    {
        GroundDetection();

        //Physics
        if (isGrounded)
        {
            UpdateSpeed();
            Turn();
            ApplyTraction();
        }
        else
        {
            ApplyTurnMomentumInAir();
        }

        AllignCarToGround();
        ApplyGravity();

        //Gameplay
        UpdateBoostGauge();


        //Debug.Log($"Current Speed: {GetCurrentSpeed()}");
        Debug.DrawRay(transform.position, rb.linearVelocity.normalized * 3f, Color.blue);
        Debug.DrawRay(transform.position, targetCarForward * 2f, Color.green);
        Debug.DrawRay(transform.position, groundNormal * 2f, Color.red);
    }

    //Adds the reward and its share to the episode total of its term
    public void AddReward(RewardTerm term, float reward)
    {
        AddReward(reward);
        episodeRewards[(int)term] += reward;
    }

    private void AddStepRewards()
    {
        AddReward(RewardTerm.Time, timePenaltyReward);

        //Progress reward: meters gained along the road splines since the last step (negative when going backwards)
        if (envTraining.Circuit)
        {
            float distanceAlongCircuit = envTraining.Circuit.GetDistanceAlongCircuit(transform);
            float progress = distanceAlongCircuit - lastDistanceAlongCircuit;
            lastDistanceAlongCircuit = distanceAlongCircuit;

            //Called every physics step, so a bigger progress is a jump of the projection (e.g. to another part of the track close to the car)
            if (Mathf.Abs(progress) <= 2.0f * maxSpeed * Time.fixedDeltaTime)
            {
                AddReward(RewardTerm.Progress, progress * progressRewardPerMeter);
            }

            AddOffTrackPenalty();
        }
    }

    //Same rule as the Simulation project: penalty growing from the road edge, episode ended far from the road
    private void AddOffTrackPenalty()
    {
        float distanceFromSpline = Vector3.Distance(envTraining.Circuit.GetClosestPointFromRoad(transform), transform.position);
        if (distanceFromSpline < roadHalfWidth)
            return;

        float ratio = 1.0f + (distanceFromSpline - roadHalfWidth) / (maxDistanceFromSpline - roadHalfWidth);
        AddReward(RewardTerm.OffTrack, -splineDistancePenaltyMultiplier * ratio / MaxStep);

        if (distanceFromSpline > maxDistanceFromSpline)
        {
            AddReward(RewardTerm.OffTrackEnd, outOfTrackPenalty);
            EvaluationTests.FinishTrial(true);
            EndEpisode();
        }
    }

    private void RecordEpisodeStats()
    {
        if (episodeStepCount > 0)
        {
            StatsRecorder stats = Academy.Instance.StatsRecorder;
            //Average brake input (0 to 1) of the previous episode, visible in TensorBoard
            stats.Add("Car/Brake Input", episodeBrakeInputSum / episodeStepCount);

            //Total of each reward term over the episode: the Reward/ curves add up to Environment/Cumulative Reward
            for (int i = 0; i < episodeRewards.Length; i++)
            {
                stats.Add("Reward/" + (RewardTerm)i, episodeRewards[i]);
            }
        }

        episodeStepCount = 0;
        episodeBrakeInputSum = 0f;
        Array.Clear(episodeRewards, 0, episodeRewards.Length);
    }

    private void GroundDetection()
    {
        Vector3 GroundDetection = isGrounded ? -groundNormal : Vector3.down;
        if (Physics.Raycast(car.transform.position, GroundDetection, out RaycastHit groundHit, groundDetectionDistance, groundLayerMask))
        {
            isGrounded = true;
            isOnRoad = groundHit.collider.gameObject.layer == layerRoad ? true : false;
            groundNormal = groundHit.normal;
        }
        else
        {
            isGrounded = false;
            isOnRoad = false;
            groundNormal = Vector3.up;
        }
    }

    private void ApplyGravity()
    {
        if (isGrounded)
        {
            rb.linearVelocity += groundNormal * Physics.gravity.y * gravityMultGrounded * Time.fixedDeltaTime;
        }
        else
        {
            rb.linearVelocity += Vector3.up * Physics.gravity.y * gravityMultInAir * Time.fixedDeltaTime;
        }
    }

    private void UpdateSpeed()
    {
        if(GetCurrentSpeed() < GetTargetSpeed())
        {
            Accelerate();
        }
        else if(GetCurrentSpeed() > GetTargetSpeed())
        {
            if(isBraking)
            {
                Brake();
            }
            else if(!isUsingDriftBoost)
            {
                Decelerate();
            }
        }

        ClampSpeed();
    }
    private float GetTargetSpeed()
    {
        float targetSpeedMultiplier = isOnRoad ? 1.0f : offRoadTargetSpeedMultiplier;

        if (IsUsingBoost())
        {
            return targetSpeedBoost * targetSpeedMultiplier;
        }
        else if (isAccelerating)
        {
            return targetSpeedAcceleration * targetSpeedMultiplier;
        }

        return 0f;
    }

    private void Accelerate()
    {
        float accelerationFactor = Mathf.Clamp01(GetCurrentSpeed() / GetTargetSpeed());
        AnimationCurve accelerationCurve = IsUsingBoost() ? accelerationBoostBySpeed : accelerationBySpeed;
        float curMaxAcceleration = IsUsingBoost() ? maxAccelerationBoost : maxAcceleration;

        float acceleration = accelerationCurve.Evaluate(accelerationFactor) * curMaxAcceleration;
        rb.linearVelocity += GetAccelerationDirection() * acceleration * Time.fixedDeltaTime;
        rb.linearVelocity = Vector3.ClampMagnitude(rb.linearVelocity, GetTargetSpeed());
    }

    private void Decelerate()
    {
        float decelerationFactor = 1.0f - Mathf.Clamp01(GetCurrentSpeed() / maxSpeed);
        float maxDecel = isOnRoad ? maxDeceleration : maxDecelerationOffRoad;
        float deceleration = decelerationBySpeed.Evaluate(decelerationFactor) * maxDecel;
        rb.linearVelocity -= GetAccelerationDirection() * deceleration * Time.fixedDeltaTime;
    }

    private void Brake()
    {
        float brakeFactor = 1.0f - Mathf.Clamp01(GetCurrentSpeed() / maxSpeed);
        float brake = brakeDecelerationBySpeed.Evaluate(brakeFactor) * maxBrakeDeceleration;
        rb.linearVelocity -= GetAccelerationDirection() * brake * Time.fixedDeltaTime;
    }

    private Vector3 GetAccelerationDirection()
    {
        return rb.linearVelocity.magnitude > 1.0f ? rb.linearVelocity.normalized : car.transform.forward;
    }

    private void ClampSpeed()
    {
        rb.linearVelocity = Vector3.ClampMagnitude(rb.linearVelocity, maxSpeed);

        if(!IsMovingForward()) //Don't allow the car to go backwards
        {
            rb.linearVelocity = Vector3.zero;
        }
    }

    private void Turn()
    {
        //Turn only rotates the Car (the velocity will be updated during ApplyTraction())
        int dir = turnInput > 0 ? 1 : -1;
        float amount = Mathf.Abs(turnInput);
        float rotation = dir * amount;

        if(isDrifting && driftDirection != 0)
        {
            amount = (driftDirection == 1) ? Remap(turnInput, -1, 1, driftInputRemap.x, driftInputRemap.y) : Remap(turnInput, -1, 1, driftInputRemap.y, driftInputRemap.x);
            rotation = driftDirection * amount;
            driftTime += Time.fixedDeltaTime;
        }
        turnMomentum = rotation * speedTurn;
        car.Rotate(car.transform.up, turnMomentum * Time.deltaTime);

        //Offset the Mesh when drifting
        float targetMeshOffsetAngle = 0f;
        if (isDrifting && driftDirection != 0)
        {
            targetMeshOffsetAngle = driftMeshOffsetAngle * Mathf.Sign(driftDirection); 
        }
        Quaternion targetMeshRotation = Quaternion.AngleAxis(targetMeshOffsetAngle, Vector3.up);
        mesh.localRotation = Quaternion.RotateTowards(mesh.localRotation, targetMeshRotation, driftMeshOffsetRotationSpeed * Time.deltaTime);
    }

    private void ApplyTraction()
    {
        if (!isGrounded) return;

        Vector3 targetVelocity = targetCarForward * GetCurrentSpeed();
        Vector3 projectedVelocity = Vector3.ProjectOnPlane(rb.linearVelocity, groundNormal);
        rb.linearVelocity = Vector3.SmoothDamp(rb.linearVelocity, projectedVelocity, ref velProjectOnGround, projectVelocityOnGroundSmoothTime, float.PositiveInfinity, Time.fixedDeltaTime);

        //Makes the velocity smoothly reach the forward vector of the Car
        smoothTimeTraction = Mathf.MoveTowards(smoothTimeTraction, isDrifting ? tractionTimeDrift : tractionTimeDefault, tractionTimeVaritationSpeed * Time.fixedDeltaTime);
        rb.linearVelocity = Vector3.SmoothDamp(rb.linearVelocity, targetVelocity, ref smoothVelTraction, smoothTimeTraction, float.PositiveInfinity, Time.fixedDeltaTime);
    }

    private float Remap(float value, float from1, float to1, float from2, float to2)
    {
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
    }

    private void ApplyTurnMomentumInAir()
    {
        turnMomentum = Mathf.MoveTowardsAngle(turnMomentum, 0f, turnDampingInAir * Time.fixedDeltaTime);
        car.transform.Rotate(Vector3.up, turnMomentum * Time.fixedDeltaTime);
        rb.linearVelocity = Quaternion.AngleAxis(turnMomentum * Time.fixedDeltaTime, Vector3.up) * rb.linearVelocity;
    }

    private bool IsMovingForward()
    {
        if (isDrifting)
            return true;

        return Vector3.Dot(rb.linearVelocity.normalized, car.transform.forward) >= 0;
    }

    private float GetCurrentSpeed()
    {
        return IsMovingForward() ? rb.linearVelocity.magnitude : -rb.linearVelocity.magnitude;
    }

    private void AllignCarToGround()
    {
        Quaternion targetCarRotation = Quaternion.FromToRotation(car.transform.up, isGrounded ? groundNormal : Vector3.up) * car.transform.rotation;
        targetCarForward = targetCarRotation * Vector3.forward;

        float allignSpeed = isGrounded ? allignRotationSpeed : allignRotationSpeedInAir;
        car.transform.rotation = Quaternion.Slerp(car.transform.rotation, targetCarRotation, allignSpeed * Time.fixedDeltaTime);
    }

    public int GetGroundID()
    {
        if(!isGrounded) return 0;
        if(isOnRoad) return 1;
        return 2;
    }

    public void OnCircuitUpdated(Circuit circuit)
    {
        if(circuit != null)
        {
            circuit.onCheckpointCrossed -= OnCheckpointReached;
        }

        checkpointManager.FillCheckpointsList();

        if (circuit != null)
        {
            circuit.onCheckpointCrossed += OnCheckpointReached;
        }
    }

    public Circuit GetCurrentCircuit()
    {
        return envTraining.Circuit;
    }

    private void OnCheckpointReached(CarMovement agent, Circuit circuit)
    {
        EvaluationTests.AddAgentCheckpoint();
    }



    public Rigidbody GetRigidbody()
    {
        return rb; 
    }

    #region Boost
    private bool IsUsingBoost()
    {
        return wantToBoost && BoostGauge > 0f;
    }

    private void UpdateBoostGauge()
    {
        if (!IsUsingBoost()) return;

        BoostGauge -= Time.fixedDeltaTime / boostDuration;
        if (BoostGauge < 0f)
        {
            BoostGauge = 0f;
            wantToBoost = false;
            RefillBoost();
        }
        else
        {
            StopRefillBoost();
        }
    }

    private void RefillBoost()
    {
        if (cor_BoostRefill == null)
        {
            cor_BoostRefill = StartCoroutine(RefillBoostGauge());
        }
    }

    private void StopRefillBoost()
    {
        if(cor_BoostRefill != null)
        {
            StopCoroutine(cor_BoostRefill);
            cor_BoostRefill = null;
        }
    }

    IEnumerator RefillBoostGauge()
    {
        yield return new WaitForSeconds(boostDelayBeforeRefill);

        float t = Mathf.Clamp01(BoostGauge) * boostRefillDuration;
        while (t < boostRefillDuration)
        {
            BoostGauge = t / boostRefillDuration;
            t += Time.deltaTime;
            yield return null;
        }

        BoostGauge = 1.0f;
        cor_BoostRefill = null;
    }
    #endregion
}
