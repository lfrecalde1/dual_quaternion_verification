from setuptools import setup

package_name = 'dual_quaternion_verification'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='fernandorecalde@uti.edu.ec',
    description='Dual quaternion log-error verification and plotting tools.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'orientation_cost_comparison = dual_quaternion_verification.orientation_cost_comparison:main',
            'make_dual_quaternion_videos = dual_quaternion_verification.make_dual_quaternion_videos:main',
        ],
    },
)
