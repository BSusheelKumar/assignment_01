// Simulated vacuum gripper for Gazebo Fortress (ignition-gazebo6).
// When enabled, attaches the nearest model whose name starts with <filter>
// if the cup point is within <attach_distance> of its box collision surface.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <string>

#include <ignition/common/Console.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/msgs/boolean.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>
#include <sdf/Box.hh>
#include <sdf/Geometry.hh>

#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Collision.hh>
#include <ignition/gazebo/components/DetachableJoint.hh>
#include <ignition/gazebo/components/Geometry.hh>
#include <ignition/gazebo/components/Model.hh>
#include <ignition/gazebo/components/Name.hh>

namespace depal
{
using namespace ignition;
using namespace ignition::gazebo;

class VacuumGripper : public System,
                      public ISystemConfigure,
                      public ISystemPreUpdate
{
 public:
  void Configure(const Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 EntityComponentManager &_ecm, EventManager &) override
  {
    Model model(_entity);
    const std::string linkName =
        _sdf->Get<std::string>("link_name", std::string("gripper_link")).first;
    this->cupOffset =
        _sdf->Get<math::Vector3d>("cup_offset", math::Vector3d(0, 0, 0.08)).first;
    this->attachDistance =
        _sdf->Get<double>("attach_distance", 0.015).first;
    this->filter = _sdf->Get<std::string>("filter", std::string("box")).first;
    const std::string enableTopic =
        _sdf->Get<std::string>("enable_topic", std::string("/vacuum/enable")).first;
    const std::string stateTopic =
        _sdf->Get<std::string>("state_topic", std::string("/vacuum/attached")).first;

    this->link = model.LinkByName(_ecm, linkName);
    if (this->link == kNullEntity)
    {
      ignerr << "[VacuumGripper] link '" << linkName << "' not found in model '"
             << model.Name(_ecm) << "'. Available links:" << std::endl;
      for (const auto &l : model.Links(_ecm))
      {
        auto *n = _ecm.Component<components::Name>(l);
        ignerr << "    " << (n ? n->Data() : std::string("?")) << std::endl;
      }
      return;
    }

    this->node.Subscribe(enableTopic, &VacuumGripper::OnEnable, this);
    this->statePub = this->node.Advertise<msgs::Boolean>(stateTopic);
    ignmsg << "[VacuumGripper] ready on link '" << linkName << "', enable topic "
           << enableTopic << std::endl;
  }

  void PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm) override
  {
    if (_info.paused || this->link == kNullEntity) return;

    if (!this->enabled.load())
    {
      if (this->joint != kNullEntity)
      {
        _ecm.RequestRemoveEntity(this->joint);
        this->joint = kNullEntity;
        this->PublishState(false);
        ignmsg << "[VacuumGripper] released" << std::endl;
      }
      return;
    }
    if (this->joint != kNullEntity) return;   // already holding a carton

    const EntityComponentManager &cecm = _ecm;
    const auto linkPose = worldPose(this->link, cecm);
    const math::Vector3d cup =
        linkPose.Pos() + linkPose.Rot().RotateVector(this->cupOffset);

    Entity bestLink = kNullEntity;
    std::string bestName;
    double bestDist = this->attachDistance;

    cecm.Each<components::Model, components::Name>(
        [&](const Entity &_modelEntity, const components::Model *,
            const components::Name *_name) -> bool
        {
          if (_name->Data().rfind(this->filter, 0) != 0) return true;
          Model m(_modelEntity);
          for (const auto &linkEntity : m.Links(cecm))
          {
            const auto pose = worldPose(linkEntity, cecm);
            for (const auto &coll :
                 cecm.ChildrenByComponents(linkEntity, components::Collision()))
            {
              const auto *geom = cecm.Component<components::Geometry>(coll);
              if (!geom || geom->Data().Type() != sdf::GeometryType::BOX) continue;
              const double d =
                  BoxDistance(pose, geom->Data().BoxShape()->Size(), cup);
              if (d < bestDist)
              {
                bestDist = d;
                bestLink = linkEntity;
                bestName = _name->Data();
              }
            }
          }
          return true;
        });

    if (bestLink != kNullEntity)
    {
      this->joint = _ecm.CreateEntity();
      _ecm.CreateComponent(this->joint,
                           components::DetachableJoint({this->link, bestLink, "fixed"}));
      this->PublishState(true);
      ignmsg << "[VacuumGripper] attached " << bestName << " (distance "
             << bestDist << " m)" << std::endl;
    }
  }

 private:
  // Distance from point p to the surface of a box (0 if on it, negative inside).
  static double BoxDistance(const math::Pose3d &pose, const math::Vector3d &size,
                            const math::Vector3d &p)
  {
    const math::Vector3d q = pose.Rot().RotateVectorReverse(p - pose.Pos());
    const math::Vector3d d(std::abs(q.X()) - size.X() / 2.0,
                           std::abs(q.Y()) - size.Y() / 2.0,
                           std::abs(q.Z()) - size.Z() / 2.0);
    const math::Vector3d out(std::max(d.X(), 0.0), std::max(d.Y(), 0.0),
                             std::max(d.Z(), 0.0));
    return out.Length() + std::min(std::max(d.X(), std::max(d.Y(), d.Z())), 0.0);
  }

  void OnEnable(const msgs::Boolean &_msg) { this->enabled = _msg.data(); }

  void PublishState(bool _v)
  {
    msgs::Boolean m;
    m.set_data(_v);
    this->statePub.Publish(m);
  }

  Entity link{kNullEntity};
  Entity joint{kNullEntity};
  math::Vector3d cupOffset{0, 0, 0.08};
  double attachDistance{0.015};
  std::string filter{"box"};
  std::atomic<bool> enabled{false};
  transport::Node node;
  transport::Node::Publisher statePub;
};
}  // namespace depal

IGNITION_ADD_PLUGIN(depal::VacuumGripper, ignition::gazebo::System,
                    depal::VacuumGripper::ISystemConfigure,
                    depal::VacuumGripper::ISystemPreUpdate)
IGNITION_ADD_PLUGIN_ALIAS(depal::VacuumGripper, "depal::VacuumGripper")